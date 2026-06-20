"""Frozen output schema: the contract for parallel work. Don't change field
names/types without bumping SCHEMA_VERSION and telling the group."""

SCHEMA_VERSION = 3

FIELDS = {
    "schema_version": int,
    "model_id": str,
    "model_revision": str,
    "quant": str,                 # "nf4" | "int8" | "none"
    "target_lang": str,
    "split": str,                 # "harmful" | "benign"
    "prompt_id": str,             # stable across languages
    "harm_tags": list,
    "s_en": float,
    "s_target": float,
    "delta": float,               # s_target - s_en
    "lp_comply_en": float,
    "lp_refuse_en": float,
    "lp_comply_target": float,
    "lp_refuse_target": float,
    "n_prompt_tokens_en": int,
    "n_prompt_tokens_target": int,
    "prefixes_verified": bool,
    "template_id": str,           # prefix-set version tag (provenance, not a key)
    "system_prompt_en": bool,     # was a system prompt used, EN side
    "system_prompt_target": bool, # was a system prompt used, target side
    "thinking_policy": str,       # "dense" | "empty_think" (reasoning models)
}

SPLIT_VALUES = {"harmful", "benign"}


def validate(rec):
    """Raise ValueError unless rec matches FIELDS exactly. Returns rec."""
    missing = set(FIELDS) - set(rec)
    extra = set(rec) - set(FIELDS)
    if missing:
        raise ValueError(f"missing fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown fields (schema frozen): {sorted(extra)}")
    for k, typ in FIELDS.items():
        v = rec[k]
        if typ is int and isinstance(v, bool):
            raise ValueError(f"{k!r} must be int, got bool")
        if typ is float and isinstance(v, int) and not isinstance(v, bool):
            rec[k] = v = float(v)
        if not isinstance(v, typ):
            raise ValueError(f"{k!r} must be {typ.__name__}, got {type(v).__name__}")
    if rec["split"] not in SPLIT_VALUES:
        raise ValueError(f"split must be in {SPLIT_VALUES}, got {rec['split']!r}")
    if rec["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"schema_version != {SCHEMA_VERSION}")
    return rec
