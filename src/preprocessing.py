# src/preprocessing.py
import re
import pandas as pd

def normalize_text(text: str) -> str:
    """
    Light, transformer-safe preprocessing:
    - lowercase
    - replace URLs -> <URL>
    - replace phone-ish numbers -> <PHONE>
    - collapse whitespace
    """
    if text is None:
        return ""

    s = str(text).lower().strip()

    # URLs
    s = re.sub(r"(http|https)://\S+|www\.\S+", "<URL>", s)

    # Phone numbers (simple heuristic)
    s = re.sub(r"\b(\+?\d[\d\-\s]{7,}\d)\b", "<PHONE>", s)

    # Remove extra whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_dataframe(df: pd.DataFrame, label_col: str, text_col: str) -> pd.DataFrame:
    """
    Returns dataframe with standardized columns:
    - label (lowercase stripped)
    - text (original)
    - text_clean (normalized)
    Drops empty texts + duplicates.
    """
    out = df[[label_col, text_col]].copy()
    out = out.rename(columns={label_col: "label", text_col: "text"})

    out["label"] = out["label"].astype(str).str.strip().str.lower()
    out["text"] = out["text"].astype(str).fillna("")

    # Drop empty text rows
    out = out[out["text"].str.strip().astype(bool)].copy()

    # Drop duplicates
    out = out.drop_duplicates(subset=["label", "text"]).copy()

    out["text_clean"] = out["text"].apply(normalize_text)
    return out


def detect_label_and_text_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Auto-detect label and text columns for common SMS spam datasets.
    """
    cols = list(df.columns)
    low_map = {c.lower(): c for c in cols}

    # Preferred names
    label_candidates = ["label", "category", "class", "target", "v1"]
    text_candidates = ["text", "message", "sms", "body", "content", "v2"]

    label_col = None
    text_col = None

    for c in label_candidates:
        if c in low_map:
            label_col = low_map[c]
            break

    for c in text_candidates:
        if c in low_map:
            text_col = low_map[c]
            break

    # If still missing, guess label as lowest unique object column
    if label_col is None:
        obj_cols = [c for c in cols if df[c].dtype == "object"]
        if not obj_cols:
            raise ValueError("Could not detect label column (no object columns).")
        label_col = sorted([(c, df[c].nunique()) for c in obj_cols], key=lambda x: x[1])[0][0]

    # Guess text as longest avg length among remaining object columns
    if text_col is None:
        obj_cols = [c for c in cols if df[c].dtype == "object" and c != label_col]
        if not obj_cols:
            raise ValueError("Could not detect text column.")
        text_col = sorted(
            [(c, df[c].fillna("").astype(str).map(len).mean()) for c in obj_cols],
            key=lambda x: x[1],
            reverse=True
        )[0][0]

    return label_col, text_col