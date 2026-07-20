from pipelinestore import ids


def test_rate_label_is_fixed_width_and_sortable():
    labels = [ids.rate_label(r) for r in (0.2, 0.4, 0.6, 0.8)]
    assert labels == ["cr020", "cr040", "cr060", "cr080"]
    assert labels == sorted(labels)


def test_compressate_id_roundtrip():
    cid = ids.compressate_id(ids.prompt_id(1234), "lexical", 0.4)
    assert cid == "p-1234__lexical__cr040"
    assert ids.parse_compressate_id(cid) == ("p-1234", "lexical", "cr040")


def test_slug_normalises():
    assert ids.slug("Lexical / Paraphrases") == "lexical-paraphrases"


def test_rate_label_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        ids.rate_label(1.5)
