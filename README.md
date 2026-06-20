# drift

Cross-lingual safety-propensity drift on IndicSafe. For each harmful prompt we
score a latent comply-vs-refuse propensity in English and in a target Indic
language and test whether the propensity shifts. Surface jailbreak rate
saturates (models refuse in both languages), so we use a logprob proxy instead.

## The proxy
For a prompt, teacher-force generic, payload-free prefixes at the assistant turn:

    s = mean_token_logprob(comply | prompt) - mean_token_logprob(refuse | prompt)
    delta = s_target - s_en          # diff-in-diff, per prompt

Higher `s` leans toward complying. The repo holds no harmful generations: only
the prompts (from IndicSafe) and the generic prefixes.

## Parallel work hinges on one frozen thing: the output schema
`src/schema.py` is the contract. Every run writes `results/<model>__<lang>.jsonl`,
one JSON object per prompt, plus a `.manifest.json` (config + git hash).
`analyze.py` globs everyone's `.jsonl` and merges by the schema. The only rule:
do not change the schema. Five machines, zero merge coordination.

## Data
`data/IndicSafe.csv`: 500 English prompts each Google-translated into 12 Indic
languages. Parallel by construction, so `prompt_id` pairs EN and target exactly,
and the benign control is built in (Harmless Control categories), parallel under
the same translation pipeline. `data.SPLITS` maps Category to harmful/benign;
Tricky Ambiguous is excluded by default.

## Confounds and where each is handled
- token-length inflation (Indic scripts) -> per-token mean logprob
- EN-vs-target length gap -> `s` is within-language; `delta` is diff-in-diff
- phrasing dependence -> average 2-4 prefixes per side
- prompt clustering -> clustered bootstrap (cluster key = category for now)
- many languages -> Benjamini-Hochberg
- capability vs safety -> benign control split; report
  `safety_drift = harmful_median_delta - benign_median_delta`
- machine-translation quality -> sits inside `delta`; the benign-vs-harmful
  diff-in-diff isolates safety-specific drift from generic MT degradation

## Stats (per model x language), in `metrics.py`
Wilcoxon signed-rank p; signed median delta; clustered bootstrap 95% CI on the
median; fraction delta>0; 1D Wasserstein (figure only); BH across languages.

## Setup (uv)
    uv sync                 # stats + data path
    uv sync --extra gpu     # adds torch/transformers/bitsandbytes (multi-GB)
GPU person installs `--extra gpu`; everyone else can run data/stats without it.
nf4 4-bit load by default so 6-8GB GPUs run the same models.

## Run
    uv run python -m src.run configs/qwen2.5-3b__bn.yaml
    uv run python -m src.analyze

## Contribution rules
1. Don't change the schema.
2. Add + native-verify prefixes in `continuations.py` before publishing a
   language; runs with `prefixes_verified=False` are pilots, not results.
3. One YAML per (model, language) in `configs/`. Drop your `.jsonl` in `results/`.
4. Pin `model_revision` to a commit before publication runs.

## Status
Stats path verified on synthetic data. GPU scoring core (`score.py`) and the
`analyze.py`/`metrics.py` summaries are stubs that raise NotImplementedError;
implement under TDD. Bengali/Hindi prefixes are unverified drafts.
