from src import data


def test_parallel_ids_match_across_langs():
    bn = {r["prompt_id"]: r["en_text"] for r in data.load("Bengali")}
    hi = {r["prompt_id"]: r["en_text"] for r in data.load("Hindi")}
    assert bn == hi  # same id -> same English source


def test_splits_filtered():
    rows = list(data.load("Bengali", splits=("benign",)))
    assert rows and all(r["split"] == "benign" for r in rows)


def test_unknown_lang_raises():
    import pytest
    with pytest.raises(ValueError):
        list(data.load("Klingon"))
