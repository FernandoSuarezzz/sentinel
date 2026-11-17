import numpy as np
import pandas as pd
import pytest

from sentinel.evaluate import breakdown, metrics_at, threshold_for_fpr


def test_threshold_respects_target_fpr():
    rng = np.random.default_rng(0)
    y = np.array([0] * 1000 + [1] * 300)
    p = np.concatenate([rng.beta(1, 8, 1000), rng.beta(8, 1, 300)])
    t = threshold_for_fpr(y, p, target_fpr=0.01)
    fpr = ((p >= t) & (y == 0)).sum() / 1000
    assert fpr <= 0.01
    # ...and it is the *lowest* such threshold: one step lower would exceed the budget.
    lower = np.sort(p[y == 0])[::-1][10]
    assert ((p >= lower) & (y == 0)).sum() / 1000 > 0.01


def test_zero_fpr_threshold_is_above_every_negative():
    y = np.array([0, 0, 1])
    p = np.array([0.2, 0.7, 0.9])
    assert threshold_for_fpr(y, p, 0.0) > 0.7


def test_metrics_at_known_confusion_matrix():
    y = np.array([1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.7, 0.2, 0.1, 0.0])
    m = metrics_at(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["fpr"] == pytest.approx(1 / 4)


def test_breakdown_per_group():
    part = pd.DataFrame({"label": [1, 1, 0, 0], "lang": ["en", "es", "en", "es"]})
    table = breakdown(part, np.array([0.9, 0.1, 0.6, 0.1]), 0.5, "lang")
    assert table.loc["en", "recall"] == 1.0 and table.loc["en", "fpr"] == 1.0
    assert table.loc["es", "recall"] == 0.0 and table.loc["es", "fpr"] == 0.0
