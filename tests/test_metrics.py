import numpy as np

from src import metrics


def test_diff_median():
    assert metrics.diff_median([1, 2, 3], [0, 0, 0]) == 2.0
    assert metrics.diff_median([0, 0], [1, 3]) == -2.0


def test_safety_drift_ci_excludes_zero_when_separated():
    rng = np.random.default_rng(0)
    harmful = rng.normal(3.0, 0.5, 300)   # big drift
    benign = rng.normal(0.0, 0.5, 300)    # no drift
    lo, hi = metrics.safety_drift_ci(harmful, benign)
    assert lo > 0 and hi > 0              # CI on the difference excludes 0


def test_safety_drift_ci_includes_zero_when_same():
    rng = np.random.default_rng(1)
    a = rng.normal(2.0, 0.5, 300)
    b = rng.normal(2.0, 0.5, 300)         # identical drift -> safety_drift ~ 0
    lo, hi = metrics.safety_drift_ci(a, b)
    assert lo < 0 < hi


def test_safety_drift_p_small_when_separated():
    rng = np.random.default_rng(2)
    harmful = rng.normal(3.0, 0.5, 200)
    benign = rng.normal(0.0, 0.5, 200)
    assert metrics.safety_drift_p(harmful, benign, n=2000) < 0.01


def test_safety_drift_p_large_when_same():
    rng = np.random.default_rng(3)
    a = rng.normal(2.0, 0.5, 200)
    b = rng.normal(2.0, 0.5, 200)
    assert metrics.safety_drift_p(a, b, n=2000) > 0.05


def test_safety_drift_p_is_bounded():
    # add-one correction: never returns exactly 0
    p = metrics.safety_drift_p([5.0] * 50, [-5.0] * 50, n=100)
    assert 0 < p <= 1


def test_bootstrap_ci_unclustered_is_stable():
    rng = np.random.default_rng(4)
    d = rng.normal(1.0, 0.3, 300)
    lo, hi = metrics.bootstrap_ci(d)
    assert lo < np.median(d) < hi
