# Running on Google Colab (GPU)

Use the self-contained `colab_run.py` script to run the scoring pipeline on Colab's free T4 GPU.

## Cell 1 — Install dependencies

```python
!pip install -q torch transformers accelerate bitsandbytes pyyaml
```

## Cell 2 — Upload data and script

```python
from google.colab import files

# Upload colab_run.py and data/IndicSafe.csv
uploaded = files.upload()
```

## Cell 3 — Run the pipeline

```python
# Edit CONFIG in colab_run.py before running if needed (model_id, quant, etc.)
# Default: Ministral-8B-Instruct-2410, nf4, Hindi
!python colab_run.py
```

Results (`.jsonl`, `.gen.jsonl`, `.manifest.json`) auto-download when complete.

## Changing models

Edit the `CONFIG` dict at the top of `colab_run.py`:

```python
CONFIG = {
    "model_id": "mistralai/Ministral-8B-Instruct-2410",  # change this
    "model_revision": "main",
    "quant": "nf4",           # nf4 (4-bit GPU) | int8 (8-bit GPU) | none (full precision)
    "target_lang": "Hindi",   # must match a key in PREFIXES and LANG_CODE
    ...
}
```

For Qwen3-style reasoning models, uncomment `thinking_policy: empty_think`.

## Output format

Identical to local `src/run.py`:

| File | Contents |
|------|----------|
| `<model>__<lang>.jsonl` | Logprob scores (schema v3) |
| `<model>__<lang>.gen.jsonl` | Generated text + heuristic compliance flags |
| `<model>__<lang>.manifest.json` | Config + git hash + row count |
