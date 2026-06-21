"""Orchestrator: config -> score -> results/<model>__<lang>.jsonl + manifest.

  uv run python -m src.run configs/qwen2.5-3b__bn.yaml
"""

import json
import subprocess
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from . import continuations

from . import data, score
from .schema import SCHEMA_VERSION, validate

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "nogit"


def lang_code(lang):
    if lang not in data.LANG_CODE:
        raise ValueError(f"no code for {lang!r}; add to data.LANG_CODE")
    return data.LANG_CODE[lang]


def run(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    lang = cfg["target_lang"]
    pref = continuations.get(lang)
    en = continuations.get("English")
    if not pref["verified"]:
        print(f"WARNING: {lang} prefixes unverified; this run must not be published.")

    sysp = cfg.get("system_prompts") or {}
    sys_en = sysp.get("English")
    sys_tgt = sysp.get(lang)

    policy = cfg.get("thinking_policy", "dense")
    tok, model = score.load_model(cfg["model_id"], cfg["model_revision"], cfg.get("quant", "nf4"))
    if score.looks_like_reasoner(tok) and policy == "dense":
        print(f"WARNING: {cfg['model_id']} looks like a reasoning model but "
              f"thinking_policy=dense; the prefix is scored as the first thinking "
              f"token. Set thinking_policy: empty_think.")

    base = f"{cfg['model_id'].split('/')[-1]}__{lang_code(lang)}"
    if cfg.get("tag"):
        base += f"__{cfg['tag']}"
    out = RESULTS / f"{base}.jsonl"
    RESULTS.mkdir(exist_ok=True)

    rows = list(data.load(lang, cfg.get("splits", ("harmful", "benign"))))
    limit = cfg.get("limit")
    if limit:
        rows = rows[:limit]
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(rows, desc=base, unit="prompt")):
            se = score.score_prompt(tok, model, row["en_text"], en["comply"], en["refuse"],
                                    sys_en, policy)
            st = score.score_prompt(tok, model, row["target_text"], pref["comply"],
                                    pref["refuse"], sys_tgt, policy)
            rec = validate({
                "schema_version": SCHEMA_VERSION,
                "model_id": cfg["model_id"],
                "model_revision": cfg["model_revision"],
                "quant": cfg.get("quant", "nf4"),
                "target_lang": lang,
                "split": row["split"],
                "prompt_id": row["prompt_id"],
                "harm_tags": [row["category"]],
                "s_en": se["s"],
                "s_target": st["s"],
                "delta": st["s"] - se["s"],
                "lp_comply_en": se["lp_comply"],
                "lp_refuse_en": se["lp_refuse"],
                "lp_comply_target": st["lp_comply"],
                "lp_refuse_target": st["lp_refuse"],
                "n_prompt_tokens_en": se["n_prompt_tokens"],
                "n_prompt_tokens_target": st["n_prompt_tokens"],
                "prefixes_verified": bool(pref["verified"] and en["verified"]),
                "template_id": row["category"],
                "system_prompt_en": sys_en is not None,
                "system_prompt_target": sys_tgt is not None,
                "thinking_policy": policy,
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n += 1

    manifest = out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps({"config": cfg, "git_hash": git_hash(),
                                    "schema_version": SCHEMA_VERSION, "n_rows": n}, indent=2))
    print(f"wrote {n} rows -> {out}")


if __name__ == "__main__":
    run(sys.argv[1])
