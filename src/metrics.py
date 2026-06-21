"""Stats over the per-prompt delta vector (one model x language)."""

import numpy as np
from scipy import stats


def wilcoxon(deltas):
    """Two-sided Wilcoxon signed-rank p-value (shift of median from 0).
    Raises ValueError if all deltas are zero (scipy)."""
    return float(stats.wilcoxon(np.asarray(deltas, float)).pvalue)


def median_delta(deltas):
    return float(np.median(deltas))


def frac_positive(deltas):
    d = np.asarray(deltas, float)
    return float(np.mean(d > 0))


def wasserstein(s_en, s_target):
    return float(stats.wasserstein_distance(s_en, s_target))


def _boot_medians(deltas, clusters, n, rng):
    """n bootstrap medians of `deltas`. Clustered: resample whole clusters."""
    d = np.asarray(deltas, float)
    if clusters is None:
        idx = rng.integers(0, len(d), size=(n, len(d)))
        return np.median(d[idx], axis=1)
    clusters = np.asarray(clusters)
    groups = [d[clusters == c] for c in np.unique(clusters)]
    k = len(groups)
    meds = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, k, size=k)
        meds[i] = np.median(np.concatenate([groups[j] for j in pick]))
    return meds


def bootstrap_ci(deltas, clusters=None, n=10000, seed=0):
    """95% percentile CI on the median. Clustered: resample clusters with
    replacement (preserves within-cluster correlation)."""
    meds = _boot_medians(deltas, clusters, n, np.random.default_rng(seed))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def safety_drift_ci(harmful, benign, harmful_clusters=None, benign_clusters=None,
                    n=10000, seed=0):
    """Point estimate, 95% CI, and two-sided bootstrap p for
    safety_drift = median(harmful) - median(benign). Resamples the two groups
    independently (clustered), so the CI reflects uncertainty in both medians.
    Returns (point, lo, hi, p)."""
    rng = np.random.default_rng(seed)
    diff = _boot_medians(harmful, harmful_clusters, n, rng) \
        - _boot_medians(benign, benign_clusters, n, rng)
    point = float(np.median(harmful) - np.median(benign))
    lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
    p = float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))
    return point, lo, hi, p


def bh(pvalues, alpha=0.05):
    """Benjamini-Hochberg. Returns list[bool] (survives) in input order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = p[order] <= thresh
    cutoff = np.max(np.where(passed)[0]) if passed.any() else -1
    out = np.zeros(m, bool)
    out[order[: cutoff + 1]] = True
    return out.tolist()


def safety_drift(harmful_median, benign_median):
    return harmful_median - benign_median
