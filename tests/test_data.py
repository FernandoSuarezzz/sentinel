import pandas as pd
import pytest

from sentinel import data, synth


def _public(rows):
    return pd.DataFrame(
        [{"text": t, "label": y, "source": "pub", "lang": "en", "family": "public", "split": s}
         for t, y, s in rows]
    )


def test_normalize_collapses_case_and_whitespace():
    assert data.normalize("  Ignore\n\tALL  rules ") == "ignore all rules"
    assert data.text_key("Hello  World") == data.text_key("hello world")


def test_clean_drops_texts_with_conflicting_labels():
    df = _public([("same text", 0, "train"), ("Same  TEXT", 1, "train"), ("other", 0, "train")])
    out, stats = data.clean(df)
    assert list(out["text"]) == ["other"]
    assert stats["label_conflicts_dropped"] == 2


def test_clean_keeps_duplicate_in_test_not_train():
    """A text in both train and test is a leak: the test copy must survive."""
    df = _public([("leaky prompt", 1, "train"), ("Leaky prompt", 1, "test"), ("x" * 5, 0, "train")])
    out, _ = data.clean(df)
    leaky = out[out["text"].str.lower() == "leaky prompt"]
    assert len(leaky) == 1 and leaky["split"].item() == "test"


def test_clean_filters_by_length():
    df = _public([("ab", 0, "train"), ("fine text", 0, "train"), ("z" * 30_000, 1, "train")])
    out, _ = data.clean(df)
    assert list(out["text"]) == ["fine text"]


def test_validation_split_is_stratified_and_disjoint():
    rows = [(f"safe text {i}", 0, "train") for i in range(200)]
    rows += [(f"attack text {i}", 1, "train") for i in range(100)]
    df = data.add_validation_split(_public(rows), val_frac=0.1)
    val = df[df["split"] == "val"]
    assert len(val) == 30
    assert val["label"].mean() == pytest.approx(1 / 3, abs=0.05)


def test_synthetic_is_deterministic():
    cfg = synth.SynthConfig(n_benign=50, n_hard_negative=50, n_indirect=50, n_direct=20, seed=7)
    pd.testing.assert_frame_equal(synth.generate(cfg), synth.generate(cfg))


def test_synthetic_holds_out_families_and_has_both_languages():
    df = synth.generate()
    train_families = set(df.loc[(df["split"] == "train") & (df["label"] == 1), "family"])
    assert train_families.isdisjoint(synth.OOD_FAMILIES)
    ood = df[df["split"] == "ood_test"]
    assert set(ood.loc[ood["label"] == 1, "family"]) == set(synth.OOD_FAMILIES)
    assert (ood["label"] == 0).any(), "OOD set needs benign rows to measure false positives"
    assert set(df["lang"]) == {"en", "es"}
    for split in ("train", "test", "ood_test"):
        assert set(df.loc[df["split"] == split, "label"]) == {0, 1}


def test_no_text_is_shared_between_training_and_evaluation():
    built, _ = data.build(_public([("filler text", 0, "train")]), synth.generate())
    seen = set(built.loc[built["split"].isin(["train", "val"]), "text"].map(data.normalize))
    for split in ("test", "ood_test"):
        held_out = set(built.loc[built["split"] == split, "text"].map(data.normalize))
        assert seen.isdisjoint(held_out)


def test_test_attacks_use_payload_variants_never_seen_in_training():
    df = synth.generate()
    direct = df[(df["label"] == 1) & (df["family"] != "public")]
    # Bare payloads (direct attacks) are the payload string itself.
    bare = direct[~direct["text"].str.contains(r"[.!?]\s+\S", regex=True)]
    train = set(bare.loc[bare["split"] == "train", "text"])
    test = set(bare.loc[bare["split"] == "test", "text"])
    assert train and test and train.isdisjoint(test)


def test_build_produces_all_splits():
    public = _public([(f"public safe {i}", 0, "train") for i in range(50)]
                     + [(f"public attack {i}", 1, "train") for i in range(50)]
                     + [(f"public test {i}", i % 2, "test") for i in range(20)])
    built, stats = data.build(public, synth.generate())
    assert set(built["split"]) == {"train", "val", "test", "ood_test"}
    assert list(built.columns) == data.COLUMNS
    assert stats["output_rows"] == len(built)


def test_to_schema_rejects_inverted_labels():
    raw = pd.DataFrame({"text": ["a", "b", "c"], "label": [1, 1, 0]})
    src = data.Source("x/y", "text", "label", 1, "en", max_positive_rate=0.5)
    with pytest.raises(ValueError, match="inverted"):
        data._to_schema(raw, src, "train")
