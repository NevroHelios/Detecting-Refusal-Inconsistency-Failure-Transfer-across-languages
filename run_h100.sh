#!/usr/bin/env bash
# Turnkey scoring run for a CUDA GPU box (tested target: a single H100, 80GB).
#
#   bash run_h100.sh                  # runs the default config list below
#   bash run_h100.sh configs/foo.yaml configs/bar.yaml   # run specific configs
#
# Produces results/<model>__<lang>[__tag].jsonl + .manifest.json, then a merged
# results/summary.csv. Re-summarizing later needs no GPU: `python -m src.analyze`.
set -euo pipefail
cd "$(dirname "$0")"

# Default work: the Gemma gap. Add configs/llama-3-8b__bn.yaml to also extend Llama.
CONFIGS=("$@")
if [ ${#CONFIGS[@]} -eq 0 ]; then
  CONFIGS=(configs/gemma-3-12b-it__hi__nosysp.yaml)
fi

echo "== GPU check =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: no nvidia-smi found. This script needs a CUDA GPU." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Hugging Face auth (Gemma + Llama are gated) ---------------------------
# Accept each model's license on its HF page first, then export a token:
#   export HF_TOKEN=hf_xxx      (or run: huggingface-cli login)
if [ -z "${HF_TOKEN:-}" ] && [ ! -f "${HOME}/.cache/huggingface/token" ]; then
  echo "WARNING: no HF_TOKEN set and no cached HF login. Gated models (Gemma/Llama)" >&2
  echo "         will 401. Run 'huggingface-cli login' or 'export HF_TOKEN=hf_...'." >&2
fi

# --- Python env ------------------------------------------------------------
# Prefer uv if present (uses the repo's pyproject); else a plain venv + pip.
if command -v uv >/dev/null 2>&1; then
  echo "== uv sync --extra gpu =="
  uv sync --extra gpu
  uv pip install tqdm                       # missing from pyproject; src.run needs it
  RUN="uv run python"
else
  echo "== venv + pip install -r requirements-h100.txt =="
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements-h100.txt
  RUN="python"
fi

# --- Score each config -----------------------------------------------------
for cfg in "${CONFIGS[@]}"; do
  echo "== scoring: ${cfg} =="
  $RUN -m src.run "${cfg}"
done

# --- Optional capability check (slow; uncomment to generate sample outputs) -
# A large drift can be an artifact of the model not reading the target language.
# src.gen writes results/<base>.gen.jsonl (a diagnostic analyze ignores).
# for cfg in "${CONFIGS[@]}"; do $RUN -m src.gen "${cfg}"; done

# --- Merge + summarize (CPU; safe to re-run anywhere) ----------------------
echo "== analyze =="
$RUN -m src.analyze

echo
echo "DONE. New result files in results/ :"
ls -1 results/*.jsonl | grep -v '\.gen\.jsonl$' || true
echo "Summary table: results/summary.csv"
echo "Send back the new results/<model>__*.jsonl + .manifest.json (and gen.jsonl if you ran src.gen)."
