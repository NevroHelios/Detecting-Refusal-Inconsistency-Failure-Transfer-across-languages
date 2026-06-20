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
- resampling unit -> bootstrap over prompts by default; `--cluster` clusters by
  harm category as a sensitivity check (few categories -> coarse CI)
- many languages -> Benjamini-Hochberg, across languages within one
  model/condition family (not across the whole table)
- capability vs safety -> benign control split. The headline estimand is
  `safety_drift = median(harmful delta) - median(benign delta)`, **tested
  directly**: a difference-of-medians bootstrap CI and a label-permutation
  p-value, both resampling prompts within each split. (The harmful-only Wilcoxon
  is reported too, but only as a secondary "did target shift vs EN at all".)
- machine-translation quality -> sits inside `delta`; the benign-vs-harmful
  diff-in-diff isolates safety-specific drift from generic MT degradation

## Stats (per model x language), in `metrics.py`
Headline: `safety_drift` with a difference-of-medians bootstrap 95% CI
(`safety_drift_ci`) and a two-sided permutation p (`safety_drift_p`), BH-corrected
across languages. Secondary/descriptive: signed harmful median delta + its
bootstrap CI, fraction delta>0, harmful Wilcoxon p, 1D Wasserstein (figure only).

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
Scoring core (`score.py`), the orchestrator (`run.py`), and the
`analyze.py`/`metrics.py` summaries are implemented; `tests/` passes. Pilot runs
exist for Qwen3-0.6B on Bengali/Hindi, but their prefixes are **unverified
drafts** (`prefixes_verified=False`) on an unpinned `model_revision`, so per
rules #2/#4 they are pilots, not publishable results.
