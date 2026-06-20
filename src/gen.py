"""Generate model outputs for N random prompts (capability diagnostic).

Decoupled from scoring so the metric run (src.run) stays fast. Reads the same
config YAML; driven by the `generate:` block:

  generate:
    n: 50              # number of random prompts to sample
    max_new_tokens: 1024
    seed: 0            # reproducible sample

  uv run python -m src.gen configs/qwen3-4b__bn.yaml

Writes results/<base>.gen.jsonl (same sidecar format src.analyze ignores).
"""

import json
import random
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

from . import data, score
from .run import RESULTS, lang_code


def run(cfg_path):
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    gen = cfg.get("generate") or {}
    n = gen.get("n")
    if not n:
        raise ValueError("config needs a generate.n (number of random samples)")
    max_new = gen.get("max_new_tokens", 1024)
    seed = gen.get("seed", 0)

    lang = cfg["target_lang"]
    policy = cfg.get("thinking_policy", "dense")
    sysp = cfg.get("system_prompts") or {}
    sys_en, sys_tgt = sysp.get("English"), sysp.get(lang)

    rows = list(data.load(lang, cfg.get("splits", ("harmful", "benign"))))
    pick = random.Random(seed).sample(rows, min(n, len(rows)))

    tok, model = score.load_model(cfg["model_id"], cfg["model_revision"], cfg.get("quant", "nf4"))

    base = f"{cfg['model_id'].split('/')[-1]}__{lang_code(lang)}"
    if cfg.get("tag"):
        base += f"__{cfg['tag']}"
    out = RESULTS / f"{base}.gen.jsonl"
    RESULTS.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in tqdm(pick, desc=base, unit="gen"):
            f.write(json.dumps({
                "prompt_id": row["prompt_id"], "split": row["split"],
                "category": row["category"], "thinking_policy": policy,
                "en_text": row["en_text"], "target_text": row["target_text"],
                "gen_en": score.generate_text(tok, model, row["en_text"], sys_en, policy, max_new),
                "gen_target": score.generate_text(tok, model, row["target_text"], sys_tgt, policy, max_new),
            }, ensure_ascii=False) + "\n")
            f.flush()
    print(f"wrote {len(pick)} generations -> {out}")


if __name__ == "__main__":
    run(sys.argv[1])
