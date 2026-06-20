import pytest

from src.schema import SCHEMA_VERSION, validate

GOOD = {
    "schema_version": SCHEMA_VERSION, "model_id": "m", "model_revision": "r",
    "quant": "nf4", "target_lang": "Bengali", "split": "harmful", "prompt_id": "p000",
    "harm_tags": ["x"], "s_en": 0.1, "s_target": 0.2, "delta": 0.1,
    "lp_comply_en": -1.0, "lp_refuse_en": -1.1, "lp_comply_target": -1.2,
    "lp_refuse_target": -1.3, "n_prompt_tokens_en": 5, "n_prompt_tokens_target": 9,
    "prefixes_verified": False, "template_id": "x",
    "system_prompt_en": False, "system_prompt_target": False,
    "thinking_policy": "dense",
}


def test_good_passes():
    assert validate(dict(GOOD)) == GOOD


def test_missing_field():
    bad = dict(GOOD); del bad["delta"]
    with pytest.raises(ValueError):
        validate(bad)


def test_unknown_field():
    bad = dict(GOOD); bad["oops"] = 1
    with pytest.raises(ValueError):
        validate(bad)


def test_bad_split():
    bad = dict(GOOD); bad["split"] = "harmless"
    with pytest.raises(ValueError):
        validate(bad)


def test_bool_not_int():
    bad = dict(GOOD); bad["n_prompt_tokens_en"] = True
    with pytest.raises(ValueError):
        validate(bad)
