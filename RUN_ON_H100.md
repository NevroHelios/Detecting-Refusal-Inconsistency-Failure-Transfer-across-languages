# Run drift scoring on an H100

Self-contained bundle to score one or more (model × language) configs on a CUDA
GPU and produce the schema-v3 result files + a merged `summary.csv`.

## What's in here
- `src/`, `data/IndicSafe.csv`, `configs/`, `tests/` — the pipeline + corpus
- `run_h100.sh` — one-command driver (install → score → analyze)
- `requirements-h100.txt` — deps for a plain `pip` box (includes `tqdm`, which
  the repo's `pyproject` is missing)
- `configs/gemma-3-12b-it__hi__nosysp.yaml` — **the main run** (Gemma 3 12B, Hindi)
- `configs/llama-3-8b__bn.yaml` — bonus: Llama-3-8B extended to Bengali

## Prereqs (do this first)
These models are **gated** on Hugging Face. On the account whose token you'll use:
1. Accept the license at https://huggingface.co/google/gemma-3-12b-it
   (and https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct if running Llama).
2. Get a token from https://huggingface.co/settings/tokens.

## Run
```bash
unzip drift-h100.zip && cd drift-h100
export HF_TOKEN=hf_xxxxxxxx          # or: huggingface-cli login
bash run_h100.sh                     # default: Gemma 3 12B, Hindi
```
To run specific / multiple configs:
```bash
bash run_h100.sh configs/gemma-3-12b-it__hi__nosysp.yaml configs/llama-3-8b__bn.yaml
```

Each config writes `results/<model>__<lang>[__tag].jsonl` and a `.manifest.json`,
then `run_h100.sh` runs `python -m src.analyze` to produce `results/summary.csv`.

### Quick smoke test first (recommended)
Uncomment `# limit: 20` in the Gemma config and run once — that scores 20
harmful + 20 benign prompts in a couple minutes and confirms auth/template/VRAM
are all fine before the full ~455-prompt run. Then comment it back out.

## Notes / gotchas
- **VRAM**: Gemma-3-12B in `nf4` is ~7–8 GB; an H100 fits it easily. The config
  uses `nf4` to match the existing Llama-8B/Qwen runs (so cross-model numbers are
  comparable). If you'd rather have bf16 full precision, set `quant: none` — fits
  fine on an H100 (~24 GB) but is a different quantization than the other rows.
- **No system prompt for Gemma**: Gemma's chat template has no `system` role, so
  there's only a `nosysp` Gemma config. (Llama/Qwen do support system prompts.)
- **Prefixes are verified**: every language in `src/continuations.py` is
  `verified=True`, so these rows are *not* prefix-pilots (rule #2 is satisfied).
  The one remaining publication caveat is rule #4: `model_revision: main` is
  unpinned — pin it to a commit hash (find it on the model's HF "Files" page)
  before treating the numbers as final.
- **Interpreting the output**: trust the `safety_drift` column (benign-controlled
  + permutation-tested), not the raw `median_delta`. Rows where `saturated=True`
  (every prompt drifted the same way) are a logprob-scale/prefix artifact in the
  raw delta — the benign-subtracted `safety_drift` is the real signal.

## Sending results back
Commit (or just zip and return) the new files:
```
results/Gemma-3-12b-it__hi__nosysp.jsonl
results/Gemma-3-12b-it__hi__nosysp.manifest.json
   (+ the Llama Bengali files if you ran that config)
```
The `.jsonl` is schema-v3, so it merges into the shared `analyze` with no
coordination. Do **not** commit `*.gen.jsonl` (model generations on harmful
prompts) — `.gitignore` already excludes them.
