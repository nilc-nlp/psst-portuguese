"""
1_inference.py
==============
Etapa 1: Inferência do modelo PSST sobre o corpus.

Lê o dataset do Hugging Face, aplica todos os filtros de frequência
configurados e guarda as predições em CSV.

Uso:
    python 1_inference.py [--output predictions.csv]

Argumentos:
    --output   Caminho do CSV de saída (default: predictions.csv)
"""

import os
import csv
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy.signal import butter, sosfilt
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

warnings.filterwarnings("ignore")
load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Etapa 1: Inferência PSST")
    parser.add_argument(
        "--output", "-o",
        default="predictions.csv",
        help="Caminho do CSV de saída (default: predictions.csv)"
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

IU_TOKEN        = "!!!!!"
MODEL_PATH      = os.getenv("MODEL_PATH", "./psst-model")
MODEL_SOURCE    = os.getenv("MODEL_SOURCE", "local")   # "local" | "hf"
SAMPLE_RATE     = 16000
MIN_DURATION    = 1.0
CHUNK_DURATION  = 30

PRECISION       = "fp32"   # "fp32" | "fp16" | "bf16"

FREQ_FILTERS = [
    None,
    ("lowpass",  200),
    ("lowpass",  400),
    ("lowpass",  800),
    ("lowpass",  1600),
    ("lowpass",  3200),
    ("highpass", 200),
    ("highpass", 400),
    ("highpass", 600),
    ("highpass", 800),
    ("highpass", 1200),
    ("highpass", 1600),
    ("highpass", 3200),
]

assert PRECISION in ("fp32", "fp16", "bf16")
_compute_dtype = (
    torch.bfloat16 if PRECISION == "bf16" and torch.cuda.is_available()
    else torch.float16 if PRECISION == "fp16" and torch.cuda.is_available()
    else torch.float32
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def text_without_iu_markers(text: str) -> str:
    return " ".join(text.replace(IU_TOKEN, " ").split())


def filter_label(filter_spec) -> str:
    if filter_spec is None:
        return "no_filter"
    ftype, fc = filter_spec
    return f"{ftype}_{fc}hz"


def apply_frequency_filter(
    audio: np.ndarray,
    sr: int,
    filter_spec,
    order: int = 5,
) -> np.ndarray:
    if filter_spec is None:
        return audio
    ftype, fc = filter_spec
    nyq = sr / 2.0
    if fc >= nyq:
        raise ValueError(f"Frequência de corte {fc}Hz >= Nyquist {nyq}Hz")
    sos = butter(order, fc / nyq, btype=ftype, output="sos")
    return sosfilt(sos, audio).astype(np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DO MODELO
# ─────────────────────────────────────────────────────────────────────────────

def load_model():
    print(f"\nDispositivo : {DEVICE}")
    print(f"Precisão    : {PRECISION}")
    print(f"\nCarregando modelo de '{MODEL_PATH}' ({MODEL_SOURCE})...")

    processor = WhisperProcessor.from_pretrained(MODEL_PATH)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=_compute_dtype,
    )
    model.to(DEVICE)
    model.eval()

    # [FIX-3]
    model.generation_config.forced_decoder_ids    = None
    model.generation_config.suppress_tokens       = []
    model.generation_config.begin_suppress_tokens = []

    # [FIX-4]
    processor.tokenizer.clean_up_tokenization_spaces = False

    print("Modelo carregado.")
    return processor, model

# ─────────────────────────────────────────────────────────────────────────────
# PRÉ-PROCESSAMENTO DO CORPUS
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_corpus(hf_dataset) -> tuple[list[np.ndarray], list[str], list]:
    """
    Retorna (audios_raw, transcripts, file_ids).
    """
    data = hf_dataset["test"]
    audios, transcripts, file_ids = [], [], []
    discarded = 0

    for example in tqdm(data, desc="Pré-processando"):
        raw = np.array(example["audio"]["array"], dtype=np.float32)
        orig_sr = example["audio"]["sampling_rate"]
        if orig_sr != SAMPLE_RATE:
            raw = librosa.resample(raw, orig_sr=orig_sr, target_sr=SAMPLE_RATE)
        duration_s = len(raw) / SAMPLE_RATE
        if duration_s > CHUNK_DURATION or duration_s <= MIN_DURATION:
            discarded += 1
            continue
        audios.append(raw)
        transcripts.append(example["Text"])
        file_ids.append(example.get("File", None))

    print(f"  Mantidos: {len(audios)} | Descartados (≤{MIN_DURATION}s ou >{CHUNK_DURATION}s): {discarded}")
    return audios, transcripts, file_ids

# ─────────────────────────────────────────────────────────────────────────────
# INFERÊNCIA
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    audios: list[np.ndarray],
    filter_spec,
    processor,
    model,
) -> list[str]:
    """
    [FIX-2] model.generate() envolto em try/except.
    """
    chunk_size = SAMPLE_RATE * CHUNK_DURATION
    predictions = []

    for audio in tqdm(audios, desc=f"Inferência [{filter_label(filter_spec)}]", leave=False):
        filtered = apply_frequency_filter(audio, SAMPLE_RATE, filter_spec)
        if len(filtered) < chunk_size:
            filtered = np.pad(filtered, (0, chunk_size - len(filtered)))

        inputs = processor(filtered, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats  = inputs.input_features.to(DEVICE).to(_compute_dtype)

        try:
            with torch.no_grad():
                generated = model.generate(feats)
            if generated.shape[-1] == 0:
                predictions.append("")
                continue
            pred_str = processor.tokenizer.decode(
                generated[0], skip_special_tokens=True
            )
        except (IndexError, RuntimeError) as exc:
            print(f"\n  [AVISO] Geração falhou ({filter_label(filter_spec)}): {exc} — usando string vazia.")
            pred_str = ""

        predictions.append(pred_str)

    return predictions

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_csv = args.output

    print("CUDA available:", torch.cuda.is_available())

    # Carregar modelo
    processor, model = load_model()

    # Carregar corpus
    print("\nCarregando corpus principal (nilc-nlp/prosodic-segmentation-dataset-updated)...")
    main_dataset = load_dataset("nilc-nlp/prosodic-segmentation-dataset-updated")
    audios, transcripts, file_ids = preprocess_corpus(main_dataset)

    # CSV fields
    FIELDS = [
        "corpus", "filter", "sample_index",
        "file_id", "gold", "pred",
        "audio_path",   # reservado para uso futuro / debug
    ]

    rows = []
    for filter_spec in FREQ_FILTERS:
        flabel = filter_label(filter_spec)
        print(f"\nFiltro: {flabel}")
        predictions = run_inference(audios, filter_spec, processor, model)

        for idx, (gold, pred, fid) in enumerate(zip(transcripts, predictions, file_ids)):
            rows.append({
                "corpus":       "main",
                "filter":       flabel,
                "sample_index": idx,
                "file_id":      fid or "",
                "gold":         gold,
                "pred":         pred,
                "audio_path":   "",
            })

    # Salvar CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPredições salvas em '{output_csv}' ({len(rows)} linhas).")


if __name__ == "__main__":
    main()