"""Load IndicSafe parallel corpus (500 EN prompts x 12 Indic langs, machine-
translated). prompt_id is a stable index over the sorted English set so EN and
target rows of a pair share an id."""

import csv
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "IndicSafe.csv"

LANG_CODE = {
    "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
    "Marathi": "mr", "Odia": "or", "Tamil": "ta", "Telugu": "te",
    "Hindi": "hi", "Urdu": "ur", "Nepali": "ne", "Punjabi": "pa",
}

SPLITS = {
    "Harmful Instructions": "harmful",
    "Offensive/Hate Speech": "harmful",
    "Misinformation/Conspiracy": "harmful",
    "Gender & Caste Discrimination": "harmful",
    "Religious Hate or Provocation": "harmful",
    "Political Manipulation Prompts": "harmful",
    "Health Misinformation": "harmful",
    "Harmless Control": "benign",
    "Harmless Control Prompts": "benign",
    "Tricky Ambiguous Prompts": "ambiguous",
}


def _rows(path=DATA_PATH):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _prompt_index(path=DATA_PATH):
    seen = sorted({r["Prompt"] for r in _rows(path)})
    w = max(3, len(str(len(seen) - 1)))
    return {p: f"p{i:0{w}d}" for i, p in enumerate(seen)}


def languages(path=DATA_PATH):
    return sorted({r["Language"] for r in _rows(path)})


def load(lang, splits=("harmful", "benign"), path=DATA_PATH):
    """Yield {prompt_id, en_text, target_text, split, category, language}."""
    idx = _prompt_index(path)
    want = set(splits)
    n = 0
    for r in _rows(path):
        if r["Language"] != lang:
            continue
        cat = r["Category"]
        if cat not in SPLITS:
            raise ValueError(f"unmapped Category {cat!r}; add to data.SPLITS")
        if SPLITS[cat] not in want:
            continue
        n += 1
        yield {
            "prompt_id": idx[r["Prompt"]],
            "en_text": r["Prompt"],
            "target_text": r["test_input"],
            "split": SPLITS[cat],
            "category": cat,
            "language": lang,
        }
    if n == 0:
        raise ValueError(f"no rows for {lang!r} with splits {want}")
