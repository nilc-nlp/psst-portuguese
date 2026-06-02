import pandas as pd
import re

MAX_DURATION = 30.0

# Padrões a remover POR COMPLETO (conteúdo + parênteses)
REMOVE_FULL_PATTERNS = [
    r"\(\(risos de L2\)\)",
    r"\(\(risos de doc 1\)\)",
    r"\(\(risos de L1\)\)",
    r"\(\(risos de doc 2\)\)",
    r"\(\(riso\)\)",
    r"\(\(risos\)\)",
    r"\(\(incompreensível\)\)",
    r"\(\(tosse\)\)",
    r"\(\(tossiu\)\)",
    r"\(\(rindo\)\)",
    r"\(\(negativamente\)\)",
    r"\(\(pigarro\)\)",
    r"\(\(clique\)\)",
    r"\(\(Doc1 dá uma risada de fundo\)\)",
    r"\(\(ruído\)\)",
    r"\(\(riu\)\)",
    r"\(\(imita barulho de carro\)\)",
    r"\(\)",
    r"\( \)",
    r":",
    r"uhn",
    r"ahn",
    r"/",
    r"\.\.\.",
    r"\. \. \.",
    r"\[",
    r"\]",
]

REMOVE_FULL_REGEX = re.compile(
    "|".join(REMOVE_FULL_PATTERNS), flags=re.IGNORECASE
)

# Só remove os parênteses, mantém o conteúdo interno
STRIP_PARENS_REGEX = re.compile(r"\(\(([^)]*)\)\)|\(([^)]*)\)")

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    # Etapa 1: remove padrões completos (termo + parênteses)
    text = REMOVE_FULL_REGEX.sub("", text)
    # Etapa 2: remove parênteses restantes, mantendo o conteúdo
    text = STRIP_PARENS_REGEX.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Etapa 3 (opcional): limpa separadores órfãos
    text = re.sub(r"(\|\|\|)+", "|||", text)  # colapsa múltiplos |||
    text = re.sub(r"^\|\|\||\|\|\|$", "", text)  # remove do início/fim
    return text


def concatenate_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index().rename(columns={"index": "orig_idx"})
    df = df.sort_values(by=["File", "Start"]).reset_index(drop=True)

    results = []

    for file_id, group in df.groupby("File", sort=False):
        current = None

        for row in group.itertuples(index=False):
            cleaned_text = clean_text(row.Text)

            if current is None:
                # Pula vazios sem segmento anterior
                if cleaned_text == "":
                    continue
                current = {
                    "File": row.File,
                    "Speaker": row.Speaker,
                    "Number_Concatenated": 1,
                    "Index_Concatenated": [row.orig_idx],
                    "Start": row.Start,
                    "End": row.End,
                    "Text": cleaned_text,
                }
                continue

            same_speaker = (row.Speaker == current["Speaker"])
            new_duration = row.End - current["Start"]
            fits = same_speaker and new_duration < MAX_DURATION

            if cleaned_text == "":
                if fits:
                    # Segmento vazio mas válido: estende o intervalo silenciosamente
                    current["End"] = row.End
                    current["Index_Concatenated"].append(row.orig_idx)
                else:
                    # Vazio e fora das regras: finaliza o atual, não inicia novo
                    current["Duration"] = current["End"] - current["Start"]
                    current["Index_Concatenated"] = ",".join(map(str, current["Index_Concatenated"]))
                    results.append(current)
                    current = None
                continue

            if fits:
                current["End"] = row.End
                current["Text"] += "|||" + cleaned_text
                current["Number_Concatenated"] += 1
                current["Index_Concatenated"].append(row.orig_idx)
            else:
                current["Duration"] = current["End"] - current["Start"]
                current["Index_Concatenated"] = ",".join(map(str, current["Index_Concatenated"]))
                results.append(current)
                current = {
                    "File": row.File,
                    "Speaker": row.Speaker,
                    "Number_Concatenated": 1,
                    "Index_Concatenated": [row.orig_idx],
                    "Start": row.Start,
                    "End": row.End,
                    "Text": cleaned_text,
                }

        # flush final
        if current is not None and current["Number_Concatenated"] > 0:
            current["Duration"] = current["End"] - current["Start"]
            current["Index_Concatenated"] = ",".join(map(str, current["Index_Concatenated"]))
            results.append(current)

    return pd.DataFrame(results)


# Usage
df = pd.read_csv("output_catna.csv")
df_out = concatenate_segments(df)

df_out = df_out[[
    "File", "Speaker", "Number_Concatenated",
    "Index_Concatenated", "Start", "End", "Duration", "Text"
]]

df_out = df_out.sort_values(by=["File", "Start"]).reset_index(drop=True)

df_out.to_csv("catna_clean.csv", index=False)