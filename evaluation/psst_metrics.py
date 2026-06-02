"""
2_difflib_metrics.py
====================
Avaliação posicional de boundaries via difflib — sem MFA.

Lê o CSV produzido pelo script 1 (predictions.csv) e calcula todas as
métricas do pipeline PSST sem necessidade de alinhamento forçado.

Estratégia de avaliação de boundaries
--------------------------------------
Em vez de forçar o alinhamento temporal com o Montreal Forced Aligner,
o script compara directamente as sequências de tokens (palavras + IU_TOKEN)
usando `difflib.ndiff`:

  - IU_TOKEN na mesma posição relativa em ambos → TP
  - IU_TOKEN só no gold                         → FN
  - IU_TOKEN só na pred                         → FP
  - gap entre duas palavras consecutivas em match
    sem IU_TOKEN em nenhum dos lados             → TN  (Opção 1)

Métricas calculadas
-------------------
  - WER / CER     (texto sem IU markers, via 'evaluate')
  - Precision / Recall / F1
  - F1 macro      (sklearn average='macro': média de F1_TB e F1_NB)
  - Accuracy      ((TP + TN) / (TP + TN + FP + FN))
  - TN            (gaps entre palavras em match sem boundary em nenhum lado)
  - Subset clean_test separado (sem excluir da avaliação global)

Uso:
    python 2_difflib_metrics.py [--input predictions.csv] [--output eval_difflib.csv]

Argumentos:
    --input    CSV de entrada (saída do script 1, default: predictions.csv)
    --output   CSV de saída com métricas (default: eval_difflib.csv)
"""

import csv
import argparse
import difflib
import collections
import warnings

import numpy as np
import evaluate
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ARGUMENTOS
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Avaliação posicional de boundaries via difflib (sem MFA)"
    )
    parser.add_argument(
        "--input", "-i",
        default="predictions.csv",
        help="CSV de entrada (saída do script 1, default: predictions.csv)"
    )
    parser.add_argument(
        "--output", "-o",
        default="eval_difflib.csv",
        help="CSV de saída com métricas (default: eval_difflib.csv)"
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

IU_TOKEN = "!!!!!"

CLEAN_TEST_FILES: list[str] = [
    "SP_DID_161.TextGrid", "SP_DID_234.TextGrid"
]

# ─────────────────────────────────────────────────────────────────────────────
# UTILITÁRIOS
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_for_metrics(text: str) -> list[str]:
    """Tokeniza preservando IU_TOKEN como token separado."""
    return text.replace(IU_TOKEN, f" {IU_TOKEN} ").split()


def text_without_iu_markers(text: str) -> str:
    return " ".join(text.replace(IU_TOKEN, " ").split())

# ─────────────────────────────────────────────────────────────────────────────
# COMPARAÇÃO POSICIONAL DE BOUNDARIES
# ─────────────────────────────────────────────────────────────────────────────

def compare_boundaries_positional(
    gold_text: str,
    pred_text: str,
) -> tuple[int, int, int]:
    """
    Compara as posições de IU_TOKEN entre gold e pred usando difflib.ndiff.
    Retorna (tp, fp, fn).
    """
    gold_tokens = tokenize_for_metrics(gold_text)
    pred_tokens = tokenize_for_metrics(pred_text)

    diff = list(difflib.ndiff(gold_tokens, pred_tokens))

    tp = fp = fn = 0
    for item in diff:
        token = item[2:]
        if item.startswith("  "):
            if token == IU_TOKEN:
                tp += 1
        elif item.startswith("- "):
            if token == IU_TOKEN:
                fn += 1
        elif item.startswith("+ "):
            if token == IU_TOKEN:
                fp += 1

    return tp, fp, fn


def boundary_binary_labels_positional(
    gold_text: str,
    pred_text: str,
) -> tuple[list[int], list[int]]:
    """
    Constrói vectores binários y_true / y_pred para cálculo de F1 macro
    (sklearn) e accuracy.

    Unidade de avaliação: posições possíveis de boundary definidas como
    GAPS entre palavras consecutivas em match no diff (Opção 1).

    Classificação de cada posição:
      - IU_TOKEN em match                              → TP  (1, 1)
      - IU_TOKEN só no gold                            → FN  (1, 0)
      - IU_TOKEN só na pred                            → FP  (0, 1)
      - gap entre duas palavras em match consecutivas
        sem IU_TOKEN em nenhum dos lados               → TN  (0, 0)

    Um gap é considerado ambíguo (não gera TN) se entre as duas palavras
    em match houver alguma palavra inserida ou apagada — nesse caso não
    se pode afirmar com certeza que não havia boundary naquele ponto.

    Algoritmo de estado:
      - last_word_matched : o token anterior em match foi uma palavra não-IU
      - iu_in_gap_*       : apareceu IU_TOKEN (gold ou pred) neste gap
      - gap_broken        : apareceu palavra inserida/apagada neste gap
    """
    gold_tokens = tokenize_for_metrics(gold_text)
    pred_tokens = tokenize_for_metrics(pred_text)

    diff = list(difflib.ndiff(gold_tokens, pred_tokens))

    y_true: list[int] = []
    y_pred: list[int] = []

    last_word_matched = False
    iu_in_gap_gold    = False
    iu_in_gap_pred    = False
    gap_broken        = False

    for item in diff:
        tag   = item[:2]
        token = item[2:]

        if tag == "  ":                              # match
            if token == IU_TOKEN:
                y_true.append(1); y_pred.append(1)  # TP
                # boundary em match fecha o gap; próximo gap começa do zero
                last_word_matched = False
                iu_in_gap_gold = iu_in_gap_pred = gap_broken = False
            else:
                # palavra em match — fechar gap anterior se limpo
                if last_word_matched and not gap_broken:
                    if not iu_in_gap_gold and not iu_in_gap_pred:
                        y_true.append(0); y_pred.append(0)  # TN
                    # se houve IU num dos lados, FP/FN já foram registados
                last_word_matched = True
                iu_in_gap_gold = iu_in_gap_pred = gap_broken = False

        elif tag == "- ":                            # só no gold
            if token == IU_TOKEN:
                y_true.append(1); y_pred.append(0)  # FN
                iu_in_gap_gold = True
                last_word_matched = False
            else:
                gap_broken = True                   # palavra apagada: gap ambíguo

        elif tag == "+ ":                            # só na pred
            if token == IU_TOKEN:
                y_true.append(0); y_pred.append(1)  # FP
                iu_in_gap_pred = True
                last_word_matched = False
            else:
                gap_broken = True                   # palavra inserida: gap ambíguo

        # linhas "? ..." do ndiff: ignorar

    if not y_true:
        return [0], [0]

    return y_true, y_pred

# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS AGREGADAS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """
    Precision, Recall e F1 com guarda para TP=FP=FN=0.

      precision = TP / (TP + FP)
      recall    = TP / (TP + FN)
      f1        = 2 * P * R / (P + R)
    """
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    _EPS = 1e-8
    precision = tp / (tp + fp + _EPS)
    recall    = tp / (tp + fn + _EPS)
    denom     = precision + recall
    f1        = 0.0 if denom == 0.0 else 2.0 * precision * recall / denom
    return precision, recall, f1


def compute_f1_variants(
    gold_texts: list[str],
    pred_texts: list[str],
) -> dict:
    """
    Calcula F1 macro (sklearn) e accuracy sobre os vectores binários de gaps.

    F1 macro  : média aritmética de F1(TB) e F1(NB), matematicamente
                idêntica ao sklearn f1_score(average='macro'):

                  F1_TB  = 2·TP / (2·TP + FP + FN)
                  F1_NB  = 2·TN / (2·TN + FN + FP)
                  macro  = (F1_TB + F1_NB) / 2

                Calculada via sklearn sobre os vectores concatenados de
                todas as amostras.

    Accuracy  : (TP + TN) / (TP + TN + FP + FN) — proporção de posições
                de gap correctamente classificadas. Os TN são gaps entre
                palavras consecutivas em match sem IU_TOKEN em nenhum
                dos lados (Opção 1).

    TN        : contagem de gaps TN nos vectores concatenados.
    """
    all_y_true: list[int] = []
    all_y_pred: list[int] = []

    for gold, pred in zip(gold_texts, pred_texts):
        y_true, y_pred = boundary_binary_labels_positional(gold, pred)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # F1 macro = sklearn average='macro'
    f1_macro_val = f1_score(
        all_y_true, all_y_pred,
        average="macro", zero_division=0
    )

    # Accuracy = (TP + TN) / total
    total   = len(all_y_true)
    correct = sum(yt == yp for yt, yp in zip(all_y_true, all_y_pred))
    accuracy = correct / total if total > 0 else 1.0

    tn = int(sum(
        1 for yt, yp in zip(all_y_true, all_y_pred) if yt == 0 and yp == 0
    ))

    return {
        "f1_macro":  round(f1_macro_val, 4),
        "accuracy":  round(accuracy,     4),
        "tn":        tn,
    }

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    input_csv  = args.input
    output_csv = args.output

    clean_test_set: set[str] = set(CLEAN_TEST_FILES)

    # ── Leitura do CSV de predições (output do script 1) ─────────────────────
    print(f"\nLendo '{input_csv}'...")
    rows_in: list[dict] = []
    with open(input_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_in.append(row)

    print(f"  {len(rows_in)} linhas lidas.")

    # ── Métricas WER/CER ─────────────────────────────────────────────────────
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # ── Agrupar por (corpus, filter) ─────────────────────────────────────────
    rows_by_group: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows_in:
        key = (r["corpus"], r["filter"])
        rows_by_group[key].append(r)

    # ── Campos CSV ───────────────────────────────────────────────────────────
    DETAIL_FIELDS = [
        "corpus", "filter", "index",
        "file_id", "in_clean_test",
        "gold", "pred",
        "gold_iu_count", "pred_iu_count",
        "tp", "fp", "fn",
        "ref_plain", "hyp_plain",
    ]

    SUMMARY_FIELDS = [
        "corpus", "filter",
        "wer", "cer",
        "precision", "recall", "f1",
        "f1_macro", "accuracy",
        "tp", "fp", "fn", "tn",
        "clean_n",
        "clean_tp", "clean_fp", "clean_fn", "clean_tn",
        "clean_wer", "clean_cer", "clean_precision", "clean_recall", "clean_f1",
        "clean_f1_macro", "clean_accuracy",
    ]

    all_detail_rows: list[dict]  = []
    all_summary_rows: list[dict] = []

    # ── Cabeçalho stdout ─────────────────────────────────────────────────────
    HDR = (
        f"{'Corpus':<8} {'Filtro':<22} "
        f"{'WER':>6} {'CER':>6} "
        f"{'P':>6} {'R':>6} {'F1':>6} "
        f"{'F1_mac':>7} {'Acc':>7}"
    )
    SEP = "─" * len(HDR)
    print(f"\n{'═'*len(HDR)}")
    print("SUMÁRIO — TODAS AS CONFIGURAÇÕES")
    print(f"{'═'*len(HDR)}")
    print(HDR)
    print(SEP)

    for (corpus_label, flabel), group_rows in rows_by_group.items():
        group_rows_sorted = sorted(group_rows, key=lambda r: int(r["sample_index"]))

        gold_texts = [r["gold"] for r in group_rows_sorted]
        pred_texts = [r["pred"] for r in group_rows_sorted]

        # WER / CER (global)
        refs = [text_without_iu_markers(g) for g in gold_texts]
        hyps = [text_without_iu_markers(p) for p in pred_texts]
        wer  = wer_metric.compute(predictions=hyps, references=refs)
        cer  = cer_metric.compute(predictions=hyps, references=refs)

        # Máscara clean_test
        in_clean_mask = [
            (r["file_id"] in clean_test_set) if r["file_id"] else False
            for r in group_rows_sorted
        ]

        # ── Por amostra ───────────────────────────────────────────────────
        tp_total = fp_total = fn_total = 0
        tp_clean = fp_clean = fn_clean = 0
        gold_texts_clean: list[str] = []
        pred_texts_clean: list[str] = []
        refs_clean: list[str] = []      # ← novo
        hyps_clean: list[str] = []      # ← novo

        for row, gold, pred, ic, ref, hyp in zip(
            group_rows_sorted, gold_texts, pred_texts, in_clean_mask, refs, hyps
        ):
            tp_i, fp_i, fn_i = compare_boundaries_positional(gold, pred)
            tp_total += tp_i
            fp_total += fp_i
            fn_total += fn_i

            if ic:
                gold_texts_clean.append(gold)
                pred_texts_clean.append(pred)
                refs_clean.append(ref)  # ← novo
                hyps_clean.append(hyp)  # ← novo
                tp_clean += tp_i
                fp_clean += fp_i
                fn_clean += fn_i

            all_detail_rows.append({
                "corpus":        corpus_label,
                "filter":        flabel,
                "index":         row["sample_index"],
                "file_id":       row["file_id"],
                "in_clean_test": int(ic),
                "gold":          gold,
                "pred":          pred,
                "gold_iu_count": gold.count(IU_TOKEN),
                "pred_iu_count": pred.count(IU_TOKEN),
                "tp":            tp_i,
                "fp":            fp_i,
                "fn":            fn_i,
                "ref_plain":     text_without_iu_markers(gold),
                "hyp_plain":     text_without_iu_markers(pred),
            })

        # ── Métricas globais ──────────────────────────────────────────────
        precision, recall, f1 = _safe_precision_recall_f1(
            tp_total, fp_total, fn_total
        )
        f1v_all  = compute_f1_variants(gold_texts, pred_texts)
        tn_total = f1v_all["tn"]

        # ── Métricas clean_test ───────────────────────────────────────────
        clean_n         = len(gold_texts_clean)
        f1v_clean       = None
        clean_wer       = None   # ← novo
        clean_cer       = None   # ← novo
        clean_precision = clean_recall = clean_f1 = None
        clean_tn        = None
        clean_accuracy  = None
        if gold_texts_clean:
            # WER / CER do subset clean                     ← novo bloco
            clean_wer = wer_metric.compute(
                predictions=hyps_clean, references=refs_clean
            )
            clean_cer = cer_metric.compute(
                predictions=hyps_clean, references=refs_clean
            )
            clean_wer = round(clean_wer, 4)
            clean_cer = round(clean_cer, 4)

            f1v_clean       = compute_f1_variants(gold_texts_clean, pred_texts_clean)
            cp, cr, cf      = _safe_precision_recall_f1(tp_clean, fp_clean, fn_clean)
            clean_precision = round(cp, 4)
            clean_recall    = round(cr, 4)
            clean_f1        = round(cf, 4)
            clean_tn        = f1v_clean["tn"]
            clean_accuracy  = f1v_clean["accuracy"]

        # ── Sumário ───────────────────────────────────────────────────────
        summary_row = {
            "corpus":          corpus_label,
            "filter":          flabel,
            "wer":             round(wer, 4),
            "cer":             round(cer, 4),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "f1":              round(f1, 4),
            "f1_macro":        f1v_all["f1_macro"],
            "accuracy":        f1v_all["accuracy"],
            "tp":              tp_total,
            "fp":              fp_total,
            "fn":              fn_total,
            "tn":              tn_total,
            "clean_n":         clean_n,
            "clean_tp":        tp_clean,
            "clean_fp":        fp_clean,
            "clean_fn":        fn_clean,
            "clean_tn":        clean_tn,
            "clean_wer":       clean_wer,        # ← novo
            "clean_cer":       clean_cer,        # ← novo
            "clean_precision": clean_precision,
            "clean_recall":    clean_recall,
            "clean_f1":        clean_f1,
            "clean_f1_macro":  f1v_clean["f1_macro"] if f1v_clean else None,
            "clean_accuracy":  clean_accuracy,
        }
        all_summary_rows.append(summary_row)

        # Print
        print(
            f"{corpus_label:<8} {flabel:<22} "
            f"{wer:>6.4f} {cer:>6.4f} "
            f"{precision:>6.4f} {recall:>6.4f} {f1:>6.4f} "
            f"{f1v_all['f1_macro']:>7.4f} "
            f"{f1v_all['accuracy']:>7.4f}"
        )
        if clean_n and f1v_clean:
            print(
                f"  → clean_test (n={clean_n:4d}): "
                f"WER={clean_wer:.4f}  CER={clean_cer:.4f}  "   # ← novo
                f"P={clean_precision:.4f}  R={clean_recall:.4f}  F1={clean_f1:.4f}  "
                f"F1_mac={f1v_clean['f1_macro']:.4f}  "
                f"Acc={clean_accuracy:.4f}  TN={clean_tn}"
            )

    print(f"{'═'*len(HDR)}\n")

    # ── Exportar CSV ─────────────────────────────────────────────────────────
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(all_detail_rows)

        f.write("\n")
        f.write("# ──── SUMÁRIO GLOBAL ────\n")
        f.write("# " + ",".join(SUMMARY_FIELDS) + "\n")
        for sr in all_summary_rows:
            row_vals = [
                sr["corpus"],
                sr["filter"],
                sr["wer"],
                sr["cer"],
                sr["precision"],
                sr["recall"],
                sr["f1"],
                sr["f1_macro"],
                sr["accuracy"],
                sr["tp"],
                sr["fp"],
                sr["fn"],
                sr["tn"],
                sr["clean_n"],
                sr["clean_tp"],
                sr["clean_fp"],
                sr["clean_fn"],
                sr["clean_tn"],
                sr["clean_wer"],        # ← novo
                sr["clean_cer"],        # ← novo
                sr["clean_precision"],
                sr["clean_recall"],
                sr["clean_f1"],
                sr["clean_f1_macro"],
                sr["clean_accuracy"],
            ]
            f.write("# " + ",".join(str(v) for v in row_vals) + "\n")

        f.write("\n")
        f.write(f"# Método de boundary    : difflib posicional (sem MFA)\n")
        f.write(f"# TN                    : gaps entre palavras em match (Opção 1)\n")
        f.write(f"# F1 macro              : sklearn average='macro' (F1_TB + F1_NB) / 2\n")
        f.write(f"# Clean-test files      : {len(clean_test_set)}\n")

    print(f"CSV salvo em '{output_csv}'.")


if __name__ == "__main__":
    main()