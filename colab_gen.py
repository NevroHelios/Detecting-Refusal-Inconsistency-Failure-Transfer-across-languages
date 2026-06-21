"""Self-contained Colab generation script (decoupled from scoring).

Samples N random prompts per language, generates model responses for both
English and target language, writes .gen.jsonl files with periodic Drive saves.

Usage:
  !pip install -q torch transformers accelerate bitsandbytes pyyaml
  from google.colab import drive; drive.mount('/content/drive')
  !python colab_gen.py
"""

# ============================================================================
# CONFIG
# ============================================================================
CONFIG = {
    "model_id": "Qwen/Qwen3-1.7B",
    "model_revision": "main",
    "quant": "nf4",
    "thinking_policy": "empty_think",
    "splits": ["harmful", "benign"],
    # "tag": "gen",
    # "system_prompts": {"English": "You are a helpful assistant."},
    "generate": {
        "n": 50,                # random prompts per language
        "max_new_tokens": 1024,
        "seed": 0,
    },
}

LANGUAGES = [
    "Bengali", "Gujarati", "Kannada", "Malayalam", "Marathi",
    "Odia", "Tamil", "Telugu", "Urdu", "Nepali", "Punjabi",
]

DATA_CSV = "IndicSafe.csv"
DRIVE_DIR = "/content/drive/MyDrive/safety_results"
SAVE_EVERY = 10  # checkpoint every N generations

# ============================================================================
# SOURCE CODE
# ============================================================================
import csv, json, os, random, shutil, subprocess, sys, time
from pathlib import Path
import torch
import yaml

POLICY_KW = {"dense": {}, "empty_think": {"enable_thinking": False}}

LANG_CODE = {
    "Bengali": "bn", "Gujarati": "gu", "Kannada": "kn", "Malayalam": "ml",
    "Marathi": "mr", "Odia": "or", "Tamil": "ta", "Telugu": "te",
    "Hindi": "hi", "Urdu": "ur", "Nepali": "ne", "Punjabi": "pa",
}

SPLITS = {
    "Harmful Instructions": "harmful", "Offensive/Hate Speech": "harmful",
    "Misinformation/Conspiracy": "harmful", "Gender & Caste Discrimination": "harmful",
    "Religious Hate or Provocation": "harmful", "Political Manipulation Prompts": "harmful",
    "Health Misinformation": "harmful", "Harmless Control": "benign",
    "Harmless Control Prompts": "benign", "Tricky Ambiguous Prompts": "ambiguous",
}


def _rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def _prompt_index(path):
    seen = sorted({r["Prompt"] for r in _rows(path)})
    w = max(3, len(str(len(seen) - 1)))
    return {p: f"p{i:0{w}d}" for i, p in enumerate(seen)}


def load_data(lang, splits=("harmful", "benign"), path=DATA_CSV):
    idx = _prompt_index(path)
    want = set(splits)
    for r in _rows(path):
        if r["Language"] != lang:
            continue
        cat = r["Category"]
        if cat not in SPLITS or SPLITS[cat] not in want:
            continue
        yield {
            "prompt_id": idx[r["Prompt"]], "en_text": r["Prompt"],
            "target_text": r["test_input"], "split": SPLITS[cat],
            "category": cat, "language": lang,
        }


def load_model(model_id, revision, quant="nf4"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    kw = dict(revision=revision, device_map="auto")
    if quant == "nf4":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    elif quant == "int8":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quant == "none":
        kw["dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if not torch.cuda.is_available():
            kw["device_map"] = "cpu"
    else:
        raise ValueError(f"unknown quant {quant!r}")
    return tok, AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()


@torch.no_grad()
def generate_text(tokenizer, model, prompt, system_prompt=None,
                  thinking_policy="dense", max_new_tokens=1024):
    msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
        + [{"role": "user", "content": prompt}]
    pids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_dict=False, **POLICY_KW[thinking_policy])
    ids = torch.tensor([pids], device=model.device)
    attn = torch.ones_like(ids)
    out = model.generate(ids, attention_mask=attn, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0, len(pids):], skip_special_tokens=True)


def _save_to_drive(local_path):
    if not DRIVE_DIR:
        return
    try:
        d = Path(DRIVE_DIR)
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, d / Path(local_path).name)
    except Exception as e:
        print(f"\n  WARNING: Drive save failed: {e}")


# ============================================================================
# RUN
# ============================================================================
def run_gen(cfg, lang, tok, model):
    gen = cfg.get("generate") or {}
    n_sample = gen.get("n", 50)
    max_new = gen.get("max_new_tokens", 1024)
    seed = gen.get("seed", 0)
    policy = cfg.get("thinking_policy", "dense")
    sysp = cfg.get("system_prompts") or {}
    sys_en, sys_tgt = sysp.get("English"), sysp.get(lang)

    lc = LANG_CODE[lang]
    base = f"{cfg['model_id'].split('/')[-1]}__{lc}"
    if cfg.get("tag"):
        base += f"__{cfg['tag']}"
    out_path = f"{base}.gen.jsonl"

    rows = list(load_data(lang, cfg.get("splits", ("harmful", "benign"))))
    pick = random.Random(seed).sample(rows, min(n_sample, len(rows)))
    print(f"  {lang}: {len(pick)} prompts sampled from {len(rows)} total")

    t0 = time.time()
    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(pick):
            rec = {
                "prompt_id": row["prompt_id"], "split": row["split"],
                "category": row["category"], "thinking_policy": policy,
                "en_text": row["en_text"], "target_text": row["target_text"],
                "gen_en": generate_text(tok, model, row["en_text"], sys_en, policy, max_new),
                "gen_target": generate_text(tok, model, row["target_text"], sys_tgt, policy, max_new),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"\r  [{i+1}/{len(pick)}] {row['prompt_id']} | {rate:.1f} gen/min", end="", flush=True)

            if (i + 1) % SAVE_EVERY == 0:
                _save_to_drive(out_path)
                print(f"  [checkpoint -> Drive]", flush=True)

    print()
    _save_to_drive(out_path)
    print(f"  wrote {len(pick)} generations -> {out_path}")


def run_all(cfg, languages):
    print(f"Loading model {cfg['model_id']} (quant={cfg.get('quant', 'nf4')})...")
    t0 = time.time()
    tok, model = load_model(cfg["model_id"], cfg["model_revision"], cfg.get("quant", "nf4"))
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    for idx, lang in enumerate(languages):
        print(f"{'='*60}")
        print(f"  [{idx+1}/{len(languages)}] {lang}")
        print(f"{'='*60}")
        run_gen(cfg, lang, tok, model)
        print(f"  [{idx+1}/{len(languages)}] {lang} complete.\n")


if __name__ == "__main__":
    positional = [a for a in sys.argv[1:] if not a.startswith("-") and a.endswith((".yaml", ".yml"))]
    cfg = yaml.safe_load(Path(positional[0]).read_text()) if positional else CONFIG
    run_all(cfg, LANGUAGES)
