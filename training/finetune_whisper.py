import os
import argparse
import shutil
from dotenv import load_dotenv
import wandb
from huggingface_hub import login

# ============================================================
# ARGUMENTOS DE LINHA DE COMANDO
# ============================================================

# Filtros individuais + modo "augmented" (original + 3 filtros concatenados)
VALID_FILTERS = ["no_filter", "lowpass_3200hz", "highpass_400hz", "highpass_600hz", "augmented"]
# Filtros aplicados quando --audio_filter augmented é selecionado
AUGMENTATION_FILTERS = ["no_filter", "lowpass_3200hz", "highpass_400hz", "highpass_600hz"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PSST Training — fine-tuning Whisper com marcadores de IU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/data-poseidon/rodrigolima/psst-model-4e-1s-difflib",
        help="Diretório local onde checkpoints e modelo final serão salvos.",
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        default="nilc-nlp/psst-portuguese-4e-1s-difflib",
        help="ID do repositório no Hugging Face Hub (ex: 'org/nome-do-modelo').",
    )
    parser.add_argument(
        "--min_duration",
        type=float,
        default=1.0,
        help="Duração mínima (em segundos) dos áudios aceitos no dataset.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size por GPU.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=4,
        help="Número de passes completos sobre os dados de treino.",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="psst-portuguese-4e-1s",
        help="Nome do run no Weights & Biases.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Precisão numérica do treino: 'fp32', 'fp16' ou 'bf16'.",
    )
    parser.add_argument(
        "--delete_output_after_push",
        action="store_true",
        default=False,
        help=(
            "Se ativado, remove o diretório de output (--output_dir) após "
            "o upload bem-sucedido para o Hugging Face Hub."
        ),
    )
    parser.add_argument(
        "--audio_filter",
        type=str,
        default="no_filter",
        choices=VALID_FILTERS,
        help=(
            "Filtro de áudio aplicado no pré-processamento. "
            "'no_filter' não aplica nenhum filtro; "
            "'lowpass_3200hz', 'highpass_400hz', 'highpass_600hz' aplicam o filtro "
            "correspondente a todos os dados de treino; "
            "'augmented' concatena o dataset original com versões filtradas por "
            "lowpass_3200hz, highpass_400hz e highpass_600hz — quadruplicando o "
            "tamanho do split de treino. Eval e test usam sempre o áudio original."
        ),
    )

    return parser.parse_args()


# ============================================================
# VARIÁVEIS DE AMBIENTE
# ============================================================

load_dotenv()

hf_token  = os.getenv("HF_TOKEN")
wandb_key = os.getenv("WANDB_API_KEY")

if not hf_token:
    raise ValueError("HF_TOKEN não encontrada.")
if not wandb_key:
    raise ValueError("WANDB_API_KEY não encontrada.")

os.environ["HF_TOKEN"] = hf_token
wandb.login(key=wandb_key)
login(token=hf_token)

import math
import difflib
import numpy as np
import torch
import librosa
from dataclasses import dataclass
from typing import Dict, List, Union, Tuple
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from datasets import load_dataset
import evaluate
import torch.distributed as dist
from sklearn.metrics import f1_score

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())

# ============================================================
# CONFIGURAÇÃO  (valores base — podem ser sobrescritos pela CLI)
# ============================================================

IU_TOKEN       = "!!!!!"
MODEL_NAME     = "openai/whisper-large-v3"
SAMPLE_RATE    = 16000
CHUNK_DURATION = 30
CHUNK_SIZE     = SAMPLE_RATE * CHUNK_DURATION
LANGUAGE       = "pt"
TASK           = "transcribe"

# Gradient accumulation — afeta o batch efetivo e portanto o nº de steps.
GRAD_ACCUM = 2

WANDB_PROJECT  = "psst-training"

# ============================================================
# FILTROS DE ÁUDIO
# ============================================================

def apply_audio_filter(audio: np.ndarray, filter_name: str) -> np.ndarray:
    """
    Aplica um filtro passa-baixa ou passa-alta ao array de áudio (mono, float32).

    Parâmetros
    ----------
    audio       : np.ndarray — sinal de áudio normalizado, shape (N,)
    filter_name : str        — um de VALID_FILTERS (exceto 'augmented',
                               que é tratado em load_and_preprocess_dataset)

    Retorna
    -------
    np.ndarray com o mesmo shape de `audio`.
    """
    if filter_name == "no_filter":
        return audio

    # librosa.effects não expõe filtros IIR diretamente; usamos scipy.
    from scipy.signal import butter, sosfilt

    nyquist = SAMPLE_RATE / 2.0

    if filter_name == "lowpass_3200hz":
        cutoff_hz = 3200.0
        sos = butter(N=4, Wn=cutoff_hz / nyquist, btype="low", output="sos")

    elif filter_name == "highpass_400hz":
        cutoff_hz = 400.0
        sos = butter(N=4, Wn=cutoff_hz / nyquist, btype="high", output="sos")

    elif filter_name == "highpass_600hz":
        cutoff_hz = 600.0
        sos = butter(N=4, Wn=cutoff_hz / nyquist, btype="high", output="sos")

    else:
        raise ValueError(
            f"Filtro '{filter_name}' desconhecido. "
            f"Opções válidas (apply_audio_filter): "
            f"{[f for f in VALID_FILTERS if f != 'augmented']}"
        )

    return sosfilt(sos, audio).astype(np.float32)


# ============================================================
# UTILITÁRIOS
# ============================================================

def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)

def wandb_log_main(metrics: dict):
    if is_main_process():
        wandb.log(metrics)

def compute_max_steps(num_train_examples: int, num_gpus: int,
                      batch_size: int, num_epochs: int) -> Tuple[int, int, int]:
    """
    Calcula max_steps para cobrir exatamente num_epochs passes sobre os dados.

    Um "step" no HF Trainer = um update de gradiente, que consome:
        effective_batch = batch_size * num_gpus * GRAD_ACCUM   exemplos

    Portanto:
        steps_per_epoch = ceil(num_train_examples / effective_batch)
        max_steps       = steps_per_epoch * num_epochs
        warmup_steps    = ~7% de max_steps (mínimo 50)

    Retorna: (effective_batch, max_steps, warmup_steps)
    """
    num_gpus        = max(num_gpus, 1)
    effective_batch = batch_size * num_gpus * GRAD_ACCUM
    steps_per_epoch = math.ceil(num_train_examples / effective_batch)
    max_steps       = steps_per_epoch * num_epochs
    warmup_steps    = max(50, round(max_steps * 0.07))
    return effective_batch, max_steps, warmup_steps

# ============================================================
# CARREGAMENTO DO DATASET
# ============================================================

def _process_split_with_filter(
    split,
    split_name: str,
    min_duration: float,
    audio_filter: str,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Processa um split HF aplicando `audio_filter` a cada exemplo.

    Parâmetros
    ----------
    split        : HuggingFace dataset split
    split_name   : nome descritivo para logging (ex: "train[no_filter]")
    min_duration : duração mínima em segundos; exemplos fora de
                   [min_duration, CHUNK_DURATION] são descartados
    audio_filter : filtro a aplicar (NÃO pode ser 'augmented' aqui)

    Retorna
    -------
    (audios, transcripts) — listas paralelas após filtragem e padding
    """
    audios            = []
    transcripts       = []
    discarded         = 0
    discarded_seconds = 0.0

    print_main(f"\nProcessando split '{split_name}' ({len(split)} exemplos, "
               f"filtro='{audio_filter}')...")

    for i, example in enumerate(split):
        if i % 100 == 0:
            print_main(f"  [{split_name}] {i}/{len(split)}...", end="\r")

        raw_audio  = np.array(example["audio"]["array"], dtype=np.float32)
        orig_sr    = example["audio"]["sampling_rate"]
        transcript = example["Text"]

        if orig_sr != SAMPLE_RATE:
            raw_audio = librosa.resample(raw_audio, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

        duration = len(raw_audio) / SAMPLE_RATE

        if duration < min_duration or duration > CHUNK_DURATION:
            discarded         += 1
            discarded_seconds += duration
            continue

        raw_audio = apply_audio_filter(raw_audio, audio_filter)

        if len(raw_audio) < CHUNK_SIZE:
            raw_audio = np.pad(raw_audio, (0, CHUNK_SIZE - len(raw_audio)))

        audios.append(raw_audio)
        transcripts.append(transcript)

    print_main(f"\n  [{split_name}] Concluído.")
    print_main(f"  [{split_name}] Exemplos mantidos:   {len(audios)}")
    print_main(f"  [{split_name}] Descartados "
               f"(< {min_duration}s ou > {CHUNK_DURATION}s): {discarded}")
    if discarded > 0:
        print_main(f"  [{split_name}] Tempo descartado: "
                   f"{discarded_seconds:.1f}s ({discarded_seconds / 60:.1f} min)")

    return audios, transcripts


def load_and_preprocess_dataset(
    min_duration: float,
    audio_filter: str,
) -> Tuple[
    List[np.ndarray], List[str],
    List[np.ndarray], List[str],
    List[np.ndarray], List[str]
]:
    print_main("\n" + "=" * 60)
    print_main("Carregando dataset nilc-nlp/prosodic-segmentation-dataset-updated...")
    print_main("=" * 60)

    dataset = load_dataset("nilc-nlp/prosodic-segmentation-dataset-updated")

    print_main(f"Splits disponíveis: {list(dataset.keys())}")
    print_main(f"Colunas disponíveis: {dataset[list(dataset.keys())[0]].column_names}")

    train_data = dataset["clean"]
    test_data  = dataset["test"]

    print_main(f"\nTotal de exemplos em 'clean' (train+eval): {len(train_data)}")
    print_main(f"Total de exemplos em 'test':               {len(test_data)}")
    print_main(f"\nModo de áudio: '{audio_filter}'")

    # ── Divisão train / eval ──────────────────────────────────────────────
    train_data  = train_data.shuffle(seed=81)
    split_idx   = int(len(train_data) * 0.95)
    train_split = train_data.select(range(split_idx))
    eval_split  = train_data.select(range(split_idx, len(train_data)))

    print_main(f"\nDivisão do split 'clean':")
    print_main(f"  Train: {len(train_split)} exemplos (95%)")
    print_main(f"  Eval:  {len(eval_split)} exemplos (5%)")

    # ── Processamento do split de treino ─────────────────────────────────
    if audio_filter == "augmented":
        # Data augmentation: original + 3 versões filtradas concatenadas.
        # Eval e test usam apenas o áudio original para avaliação limpa.
        print_main("\n" + "-" * 60)
        print_main("Modo AUGMENTED: concatenando original + lp3200hz + hp400hz + hp600hz")
        print_main("-" * 60)

        train_audios: List[np.ndarray] = []
        train_transcripts: List[str]   = []

        for filt in AUGMENTATION_FILTERS:
            filt_audios, filt_transcripts = _process_split_with_filter(
                train_split,
                split_name=f"train[{filt}]",
                min_duration=min_duration,
                audio_filter=filt,
            )
            train_audios.extend(filt_audios)
            train_transcripts.extend(filt_transcripts)

        print_main(f"\nTotal de exemplos de treino após augmentation: {len(train_audios)} "
                   f"(≈ {len(AUGMENTATION_FILTERS)}× o original)")

    else:
        # Modo simples: aplica um único filtro (incluindo no_filter)
        train_audios, train_transcripts = _process_split_with_filter(
            train_split,
            split_name="train",
            min_duration=min_duration,
            audio_filter=audio_filter,
        )

    # ── Eval e test: sempre com áudio original ────────────────────────────
    eval_filter = "no_filter"
    test_filter = "no_filter"

    eval_audios, eval_transcripts = _process_split_with_filter(
        eval_split,
        split_name="eval",
        min_duration=min_duration,
        audio_filter=eval_filter,
    )
    test_audios, test_transcripts = _process_split_with_filter(
        test_data,
        split_name="test",
        min_duration=min_duration,
        audio_filter=test_filter,
    )

    print_main("\n" + "=" * 60)
    print_main("Resumo final do dataset:")
    print_main(f"  Train: {len(train_audios)} exemplos"
               + (" (4× augmented)" if audio_filter == "augmented" else ""))
    print_main(f"  Eval:  {len(eval_audios)} exemplos  (no_filter)")
    print_main(f"  Test:  {len(test_audios)} exemplos  (no_filter)")
    print_main("=" * 60 + "\n")

    return (train_audios, train_transcripts,
            eval_audios,  eval_transcripts,
            test_audios,  test_transcripts)


# ============================================================
# DATASET PYTORCH
# ============================================================

class IUDataset(torch.utils.data.Dataset):
    def __init__(self, audio_chunks: List[np.ndarray], transcripts: List[str],
                 processor: WhisperProcessor):
        self.audio_chunks = audio_chunks
        self.transcripts  = transcripts
        self.processor    = processor

    def __len__(self):
        return len(self.audio_chunks)

    def __getitem__(self, idx):
        inputs = self.processor(
            self.audio_chunks[idx],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        # text_target ativa o modo correto do tokenizer do Whisper, gerando o
        # prefixo <|startoftranscript|><|pt|><|transcribe|><|notimestamps|>
        labels = self.processor.tokenizer(
            text_target=self.transcripts[idx],
            return_tensors="pt",
        ).input_ids

        return {
            "input_features": inputs.input_features.squeeze(0),
            "labels":         labels.squeeze(0),
        }


# ============================================================
# DATA COLLATOR
# ============================================================

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        label_features = [{"input_ids":      f["labels"]}         for f in features]

        batch        = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove o BOS (<|startoftranscript|>) das labels.
        #
        # WhisperForConditionalGeneration.forward() faz shift-right internamente,
        # que prepend o decoder_start_token_id nas labels. Se as labels já começam
        # com BOS (gerado por text_target), o decoder recebe BOS duplicado, o que
        # quebra o alinhamento posicional → loss/grad NaN desde o step 1.
        bos_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if (labels[:, 0] == bos_id).all():
            labels = labels[:, 1:]
        elif (labels[:, 0] == -100).any():
            if (labels[:, 0] == bos_id).any():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

# ============================================================
# BOUNDARY HELPERS  (difflib-based, mirrors 2_difflib_metrics.py)
# ============================================================

def _tokenize_with_iu(text: str) -> List[str]:
    """Insere espaços ao redor do IU_TOKEN antes de split() para garantir
    que ele seja tratado como token independente mesmo sem espaços adjacentes."""
    return text.replace(IU_TOKEN, f" {IU_TOKEN} ").split()


def boundary_binary_labels_positional(
    gold_text: str,
    pred_text: str,
) -> Tuple[List[int], List[int]]:
    """
    Constrói vectores binários y_true / y_pred alinhados por difflib.ndiff.

      - match de IU_TOKEN      → y_true=1, y_pred=1  (TP)
      - IU_TOKEN só no gold    → y_true=1, y_pred=0  (FN)
      - IU_TOKEN só na pred    → y_true=0, y_pred=1  (FP)
      - match de palavra comum → y_true=0, y_pred=0  (TN)

    Palavras inseridas/apagadas que não são IU_TOKEN são ignoradas —
    não representam posições de boundary.

    Se não há nenhuma entrada gerada, retorna [0], [0].
    """
    gold_tokens = _tokenize_with_iu(gold_text)
    pred_tokens = _tokenize_with_iu(pred_text)

    diff = list(difflib.ndiff(gold_tokens, pred_tokens))

    y_true: List[int] = []
    y_pred: List[int] = []

    for item in diff:
        token = item[2:]
        if item.startswith("  "):
            if token == IU_TOKEN:
                y_true.append(1); y_pred.append(1)   # TP
            else:
                y_true.append(0); y_pred.append(0)   # TN
        elif item.startswith("- "):
            if token == IU_TOKEN:
                y_true.append(1); y_pred.append(0)   # FN
        elif item.startswith("+ "):
            if token == IU_TOKEN:
                y_true.append(0); y_pred.append(1)   # FP
        # linhas '? ...' são hints do ndiff — ignorar

    if not y_true:
        return [0], [0]

    return y_true, y_pred


def text_without_iu_markers(text: str) -> str:
    """Remove o IU_TOKEN e normaliza espaços extras."""
    return " ".join(t for t in text.split() if t != IU_TOKEN)

# ============================================================
# MÉTRICAS  (f1 idêntico ao f1_binary do 2_difflib_metrics.py)
# ============================================================

def compute_iu_metrics(pred, processor):
    wer_metric = evaluate.load("wer")

    predictions = pred.predictions
    labels      = pred.label_ids

    labels = np.where(labels != -100, labels, processor.tokenizer.pad_token_id)

    decoded_preds  = processor.tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = processor.tokenizer.batch_decode(labels,      skip_special_tokens=True)

    # ── WER calculado sobre o texto *sem* marcadores IU ──────────────────
    refs_plain = [text_without_iu_markers(s) for s in decoded_labels]
    hyps_plain = [text_without_iu_markers(s) for s in decoded_preds]
    wer = wer_metric.compute(predictions=hyps_plain, references=refs_plain)

    # ── F1 binary via difflib (idêntico a 2_difflib_metrics.py) ──────────
    # Concatena os vectores binários de todas as amostras e calcula
    # f1_score(..., average="binary"), que é o f1_binary do script de eval.
    all_y_true: List[int] = []
    all_y_pred: List[int] = []

    for gold_str, pred_str in zip(decoded_labels, decoded_preds):
        y_true, y_pred = boundary_binary_labels_positional(gold_str, pred_str)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    iu_f1 = float(f1_score(all_y_true, all_y_pred,
                            pos_label=1, average="binary", zero_division=0))

    # ── TP / FP / FN para logging detalhado no wandb ─────────────────────
    tp = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 1 and yp == 0)

    metrics = {
        "wer":    wer,
        "iu_f1":  iu_f1,
        "iu_tp":  tp,
        "iu_fp":  fp,
        "iu_fn":  fn,
    }

    wandb_log_main(metrics)
    return metrics

# ============================================================
# TRAINER CUSTOMIZADO
# ============================================================

class PSSTrainer(Seq2SeqTrainer):

    def __init__(self, *args, compute_dtype=torch.float32, **kwargs):
        super().__init__(*args, **kwargs)
        self._compute_dtype = compute_dtype

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Cast explícito: o feature extractor sempre retorna fp32; o modelo
        # pode estar em fp16/bf16. Sem esse cast, a conv1 do encoder quebra.
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self._compute_dtype)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)

        if is_main_process() and self.state.global_step % self.args.logging_steps == 0:
            wandb_log_main({
                "train/loss":          loss.item(),
                "train/step":          self.state.global_step,
                "train/learning_rate": (
                    self.lr_scheduler.get_last_lr()[0]
                    if self.lr_scheduler
                    else self.args.learning_rate
                ),
            })

        return (loss, outputs) if return_outputs else loss

    def log_sample_predictions(self, dataset, num_samples=5):
        if not is_main_process():
            return

        self.model.eval()
        table = wandb.Table(columns=["gold_transcript", "pred_transcript", "match"])

        for i in range(min(num_samples, len(dataset))):
            sample         = dataset[i]
            input_features = (
                sample["input_features"]
                .unsqueeze(0)
                .to(self.model.device)
                .to(self._compute_dtype)
            )

            with torch.no_grad():
                generated = self.model.generate(input_features)

            pred_str = self.processing_class.tokenizer.decode(generated[0],
                                                              skip_special_tokens=True)
            gold_str = self.processing_class.tokenizer.decode(
                [t for t in sample["labels"] if t != -100], skip_special_tokens=True
            )

            table.add_data(gold_str, pred_str, gold_str == pred_str)

        wandb.log({"sample_predictions": table})


# ============================================================
# SANITY CHECK
# ============================================================

def sanity_check(train_dataset, processor):
    print_main("\n" + "-" * 60)
    print_main("SANITY CHECK")

    sample  = train_dataset[0]
    decoded = processor.tokenizer.decode(sample["labels"], skip_special_tokens=False)

    print_main(f"  input_features shape : {sample['input_features'].shape}")
    print_main(f"  labels shape         : {sample['labels'].shape}")
    print_main(f"  labels (decoded)     : {decoded[:200]}...")

    bos_id       = processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
    pt_id        = processor.tokenizer.convert_tokens_to_ids("<|pt|>")
    first_id     = sample["labels"][0].item()

    print_main(f"  Primeiro token é BOS : {first_id == bos_id}  (id={first_id}, bos_id={bos_id})")
    print_main(f"  Contém <|pt|>        : {pt_id in sample['labels'].tolist()}")
    print_main(f"  Contém '{IU_TOKEN}'  : {IU_TOKEN in decoded}")

    collator    = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    batch       = collator([sample])
    first_label = batch["labels"][0][batch["labels"][0] != -100][0].item()
    print_main(f"  Primeiro label após collator: {first_label} "
               f"({processor.tokenizer.decode([first_label])})")

    assert first_label != bos_id, \
        "ERRO: DataCollator não removeu o BOS! Fix B não está funcionando."
    print_main("  [OK] DataCollator removeu o BOS corretamente.")
    print_main("-" * 60 + "\n")


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()

    # ── Configurações vindas da CLI (ou defaults) ──────────────────────────
    OUTPUT_DIR              = args.output_dir
    HF_REPO_ID              = args.hf_repo_id
    MIN_DURATION            = args.min_duration
    BATCH_SIZE              = args.batch_size
    NUM_EPOCHS              = args.num_epochs
    WANDB_RUN_NAME          = args.wandb_run_name
    PRECISION               = args.precision
    DELETE_OUTPUT_AFTER_PUSH = args.delete_output_after_push
    AUDIO_FILTER            = args.audio_filter

    PUSH_TO_HUB     = True
    HF_PRIVATE_REPO = False

    _use_fp16 = (PRECISION == "fp16") and torch.cuda.is_available()
    _use_bf16 = (PRECISION == "bf16") and torch.cuda.is_available()
    _compute_dtype = (
        torch.bfloat16 if _use_bf16
        else torch.float16 if _use_fp16
        else torch.float32
    )

    print_main("\n" + "=" * 60)
    print_main("PSST Training — iniciando")
    print_main("=" * 60)

    num_gpus = torch.cuda.device_count()
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print_main(f"Dispositivo principal  : {device}")
    print_main(f"Número de GPUs         : {num_gpus}")
    print_main(f"Precisão numérica      : {PRECISION}  (fp16={_use_fp16}, bf16={_use_bf16})")
    print_main(f"Filtro/modo de áudio   : {AUDIO_FILTER}"
               + (" → treino 4× (original + lp3200 + hp400 + hp600)"
                  if AUDIO_FILTER == "augmented" else ""))
    print_main(f"Output dir             : {OUTPUT_DIR}")
    print_main(f"HF repo                : {HF_REPO_ID}")
    print_main(f"Apagar output após push: {DELETE_OUTPUT_AFTER_PUSH}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print_main(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({vram:.1f} GB VRAM)")

    # wandb.init antes do dataset para que o config seja atualizado depois
    if is_main_process():
        wandb.init(
            project=WANDB_PROJECT,
            name=WANDB_RUN_NAME,
            config={
                "model":                       MODEL_NAME,
                "language":                    LANGUAGE,
                "task":                        TASK,
                "iu_token":                    IU_TOKEN,
                "precision":                   PRECISION,
                "num_epochs":                  NUM_EPOCHS,
                "sample_rate":                 SAMPLE_RATE,
                "chunk_duration":              CHUNK_DURATION,
                "batch_size_per_gpu":          BATCH_SIZE,
                "gradient_accumulation_steps": GRAD_ACCUM,
                "num_gpus":                    max(num_gpus, 1),
                "learning_rate":               1e-5,
                "min_duration":                MIN_DURATION,
                "audio_filter":                AUDIO_FILTER,
                "augmentation_filters": (
                    AUGMENTATION_FILTERS if AUDIO_FILTER == "augmented" else None
                ),
                "dataset":                     "nilc-nlp/prosodic-segmentation-dataset",
            }
        )
        print_main("wandb inicializado com sucesso.")

    # --------------------------------------------------------
    # Dataset — carregado antes de configurar o Trainer para que
    # max_steps seja calculado com base no tamanho real do split
    # de treino *após* remoção de áudios fora do intervalo de duração.
    # --------------------------------------------------------
    (train_audios, train_transcripts,
     eval_audios,  eval_transcripts,
     test_audios,  test_transcripts) = load_and_preprocess_dataset(
        min_duration=MIN_DURATION,
        audio_filter=AUDIO_FILTER,
    )

    # --------------------------------------------------------
    # Cálculo dinâmico de steps
    # --------------------------------------------------------
    num_train = len(train_audios)
    effective_batch, max_steps, warmup_steps = compute_max_steps(
        num_train, num_gpus, BATCH_SIZE, NUM_EPOCHS
    )

    # Salva/avalia ~8 vezes por run (a cada 12.5% do treino)
    checkpoint_steps = max(1, round(max_steps / 8))
    logging_steps    = max(1, checkpoint_steps // 10)

    print_main("\n" + "=" * 60)
    print_main("Configuração de steps (calculada dinamicamente):")
    print_main(f"  Exemplos de treino (após filtro de duração) : {num_train}"
               + (" (4× augmented)" if AUDIO_FILTER == "augmented" else ""))
    print_main(f"  Batch efetivo                               : {effective_batch}  "
               f"({BATCH_SIZE} × {max(num_gpus,1)} GPU × {GRAD_ACCUM} grad_accum)")
    print_main(f"  Steps por época                             : {math.ceil(num_train / effective_batch)}")
    print_main(f"  Épocas alvo                                 : {NUM_EPOCHS}")
    print_main(f"  max_steps                                   : {max_steps}")
    print_main(f"  warmup_steps                                : {warmup_steps}  (~7% do total)")
    print_main(f"  save_steps / eval_steps                     : {checkpoint_steps}  (~12.5% do treino)")
    print_main(f"  logging_steps                               : {logging_steps}")
    print_main("=" * 60 + "\n")

    if is_main_process():
        wandb.config.update({
            "num_train_examples": num_train,
            "effective_batch":    effective_batch,
            "max_steps":          max_steps,
            "warmup_steps":       warmup_steps,
            "checkpoint_steps":   checkpoint_steps,
        })

    # --------------------------------------------------------
    # Processor e modelo
    # --------------------------------------------------------
    print_main("Carregando processor e modelo Whisper...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME,
        language=LANGUAGE,
        task=TASK,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=_compute_dtype,   # evita mismatch na conv1 (fp32 input vs fp16 weights)
    )
    print_main("Processor e modelo carregados com sucesso.")

    # Forçar tokens de decodificação para PT
    forced_ids = processor.get_decoder_prompt_ids(language=LANGUAGE, task=TASK)
    model.config.forced_decoder_ids            = forced_ids
    model.config.suppress_tokens               = []
    model.generation_config.forced_decoder_ids = forced_ids
    # suppress_tokens e begin_suppress_tokens vazios garantem que o token
    # IU (!!!!! = id 28493) não seja bloqueado durante o .generate()
    model.generation_config.suppress_tokens       = []
    model.generation_config.begin_suppress_tokens = []
    print_main(f"forced_decoder_ids: {forced_ids}")

    # Adicionar token IU ao vocabulário
    num_added   = processor.tokenizer.add_tokens([IU_TOKEN])
    model.resize_token_embeddings(len(processor.tokenizer))
    iu_token_id = processor.tokenizer.convert_tokens_to_ids(IU_TOKEN)

    print_main(f"\nToken '{IU_TOKEN}' registrado.")
    print_main(f"  Tokens adicionados    : {num_added}")
    print_main(f"  Token ID              : {iu_token_id}")
    print_main(f"  Tamanho do vocabulário: {len(processor.tokenizer)}")

    if is_main_process():
        wandb.config.update({"iu_token_id": iu_token_id})

    # --------------------------------------------------------
    # Datasets PyTorch
    # --------------------------------------------------------
    print_main("\nCriando datasets PyTorch...")
    train_dataset = IUDataset(train_audios, train_transcripts, processor)
    eval_dataset  = IUDataset(eval_audios,  eval_transcripts,  processor)
    test_dataset  = IUDataset(test_audios,  test_transcripts,  processor)

    print_main(f"  Train dataset: {len(train_dataset)} exemplos")
    print_main(f"  Eval  dataset: {len(eval_dataset)}  exemplos")
    print_main(f"  Test  dataset: {len(test_dataset)}  exemplos")

    sanity_check(train_dataset, processor)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # --------------------------------------------------------
    # Argumentos de treino
    # --------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir                   = OUTPUT_DIR,
        per_device_train_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps  = GRAD_ACCUM,
        learning_rate                = 1e-5,
        warmup_steps                 = warmup_steps,
        max_steps                    = max_steps,
        max_grad_norm                = 1.0,
        fp16                         = _use_fp16,
        bf16                         = _use_bf16,
        predict_with_generate        = True,
        generation_max_length        = 225,
        generation_num_beams         = 1,
        save_steps                   = checkpoint_steps,
        eval_steps                   = checkpoint_steps,
        logging_steps                = logging_steps,
        eval_strategy                = "steps",
        save_strategy                = "steps",
        save_total_limit             = 1,    # mantém apenas 1 checkpoint no disco (~3 GB)
        load_best_model_at_end       = True,
        metric_for_best_model        = "iu_f1",
        greater_is_better            = True,
        report_to                    = "wandb",
        run_name                     = WANDB_RUN_NAME,
        ddp_find_unused_parameters   = False,
        dataloader_num_workers       = 4,
        dataloader_pin_memory        = True,
        remove_unused_columns        = False,
        push_to_hub                 = PUSH_TO_HUB,
        hub_model_id                = HF_REPO_ID,
        hub_private_repo            = HF_PRIVATE_REPO,
        hub_strategy                = "end",
        hub_token                   = hf_token,
    )

    trainer = PSSTrainer(
        model             = model,
        args              = training_args,
        train_dataset     = train_dataset,
        eval_dataset      = eval_dataset,
        data_collator     = data_collator,
        processing_class  = processor,
        compute_metrics   = lambda pred: compute_iu_metrics(pred, processor),
        compute_dtype     = _compute_dtype,
    )

    if is_main_process():
        wandb.watch(model, log="gradients", log_freq=50)
        print_main("wandb.watch ativo.")

    print_main("\n" + "=" * 60)
    print_main("Iniciando treino...")
    print_main("=" * 60 + "\n")
    trainer.train()
    print_main("\nTreino concluído!")

    # --------------------------------------------------------
    # Avaliação no test set
    # --------------------------------------------------------
    print_main("\n" + "=" * 60)
    print_main("Avaliando no test set...")
    print_main("=" * 60)
    test_results = trainer.evaluate(eval_dataset=test_dataset)

    print_main("Resultados no test set:")
    for k, v in test_results.items():
        print_main(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    wandb_log_main({f"test/{k}": v for k, v in test_results.items()})

    if is_main_process():
        print_main("\nLogando exemplos de predição no wandb...")
        trainer.log_sample_predictions(eval_dataset)
        print_main("Exemplos logados.")

        print_main(f"\nSalvando modelo em '{OUTPUT_DIR}'...")
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        print_main("Modelo salvo.")

        if PUSH_TO_HUB:
            print_main("\nEnviando modelo para o Hugging Face Hub...")
            trainer.push_to_hub(
                commit_message="Final model upload"
            )
            processor.push_to_hub(
                HF_REPO_ID,
                token=hf_token
            )
            print_main("Upload concluído!")

            # ── Remoção opcional do diretório de output ──────────────────
            if DELETE_OUTPUT_AFTER_PUSH:
                print_main(f"\nApagando diretório de output '{OUTPUT_DIR}'...")
                try:
                    shutil.rmtree(OUTPUT_DIR)
                    print_main(f"Diretório '{OUTPUT_DIR}' removido com sucesso.")
                except Exception as exc:
                    print_main(
                        f"[AVISO] Não foi possível remover '{OUTPUT_DIR}': {exc}"
                    )

        wandb.finish()

    print_main("\nTreinamento finalizado.")


if __name__ == "__main__":
    main()