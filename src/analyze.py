"""Merge every results/*.jsonl by the frozen schema -> summary table.

  uv run python -m src.analyze

Runs entirely off saved logprobs/deltas; the model is never loaded here. Per
(model, lang): wilcoxon p, median delta (harmful/benign), bootstrap CI, frac
positive, wasserstein, safety_drift; then Benjamini-Hochberg across languages."""

import glob
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from . import metrics
from .schema import validate

RESULTS = Path(__file__).resolve().parent.parent / "results"


def load_all():
    recs = []
    for fp in glob.glob(str(RESULTS / "*.jsonl")):
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(validate(json.loads(line)))
    return recs


def group(recs):
    g = defaultdict(lambda: defaultdict(list))
    for r in recs:
        key = (r["model_id"], r["target_lang"], r["system_prompt_target"],
               r["thinking_policy"])
        g[key][r["split"]].append(r)
    return g


def summarize(recs=None):
    recs = load_all() if recs is None else recs
    if not recs:
        raise ValueError(f"no result files in {RESULTS}")
    rows = []
    for (model_id, lang, sysprompt, policy), splits in group(recs).items():
        h = splits.get("harmful", [])
        b = splits.get("benign", [])
        if not h:
            continue
        dh = [r["delta"] for r in h]
        lo, hi = metrics.bootstrap_ci(dh, clusters=[r["template_id"] for r in h])
        med_h = metrics.median_delta(dh)
        med_b = metrics.median_delta([r["delta"] for r in b]) if b else float("nan")
        rows.append({
            "model": model_id.split("/")[-1],
            "lang": lang,
            "sysprompt": sysprompt,
            "thinking": policy,
            "n_harmful": len(h),
            "n_benign": len(b),
            "median_delta": round(med_h, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "frac_pos": round(metrics.frac_positive(dh), 3),
            "wilcoxon_p": metrics.wilcoxon(dh),
            "wasserstein": round(metrics.wasserstein(
                [r["s_en"] for r in h], [r["s_target"] for r in h]), 4),
            "safety_drift": round(metrics.safety_drift(med_h, med_b), 4),
            "verified": all(r["prefixes_verified"] for r in h),
        })
    df = pd.DataFrame(rows)
    df["bh_significant"] = metrics.bh(df["wilcoxon_p"].tolist())
    df["wilcoxon_p"] = df["wilcoxon_p"].round(5)
    return df.sort_values(["model", "lang", "sysprompt", "thinking"]).reset_index(drop=True)


if __name__ == "__main__":
    df = summarize()
    print(df.to_string(index=False))
    out = RESULTS / "summary.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")
