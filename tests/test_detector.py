import pytest

from sentinel.detector import InputTooLongError, chunk_ids


def test_short_input_is_a_single_window():
    assert chunk_ids([1, 2, 3], window=5, stride=2) == [[1, 2, 3]]


def test_windows_overlap_and_cover_everything():
    ids = list(range(20))
    chunks = chunk_ids(ids, window=8, stride=3)
    assert all(len(c) <= 8 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a[-3:] == b[:3]  # consecutive windows share `stride` tokens
    assert sorted({i for c in chunks for i in c}) == ids
    assert chunks[-1][-1] == ids[-1]


def test_window_must_exceed_stride():
    with pytest.raises(ValueError):
        chunk_ids([1, 2, 3], window=2, stride=2)


def test_benign_text_is_safe(detector):
    r = detector.scan("the food was great and the staff friendly")
    assert not r.flagged and r.score < 0.5 and r.flagged_chunk is None


def test_attack_is_flagged(detector):
    r = detector.scan("nice place ATTACK")
    assert r.flagged and r.score > 0.9 and r.flagged_chunk == 0


def test_attack_hidden_after_padding_is_still_caught(detector):
    """Evasion by padding: a truncating classifier would never see the last words."""
    padding = " ".join(["lovely"] * 60)  # far beyond one 16-token window
    r = detector.scan(f"{padding} ATTACK")
    assert r.chunks_scanned > 1
    assert r.flagged and r.flagged_chunk == r.chunks_scanned - 1


def test_batch_scores_each_text_independently(detector):
    results = detector.scan_batch(["good food", "ATTACK now", " ".join(["ok"] * 40)])
    assert [r.flagged for r in results] == [False, True, False]
    assert results[2].chunks_scanned > 1


def test_oversized_input_is_rejected(detector):
    with pytest.raises(InputTooLongError):
        detector.scan(" ".join(["word"] * 500))


def test_real_loading_path_from_directory():
    """Detector.from_dir with the real `tokenizers` library (installed in CI)."""
    pytest.importorskip("tokenizers")
    from conftest import TINY_MODEL

    from sentinel.detector import Detector

    d = Detector.from_dir(TINY_MODEL)
    assert d.tokenizer.encode_ids("the food was great") == [10, 11, 12, 13]
    assert not d.scan("the food was great").flagged
    assert d.scan("the food was great ATTACK").flagged
