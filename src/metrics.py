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


def bootstrap_ci(deltas, clusters=None, n=10000, seed=0):
    """95% percentile CI on the median. Clustered: resample clusters with
    replacement (preserves within-cluster correlation)."""
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, float)
    if clusters is None:
        idx = rng.integers(0, len(d), size=(n, len(d)))
        meds = np.median(d[idx], axis=1)
    else:
        clusters = np.asarray(clusters)
        groups = [d[clusters == c] for c in np.unique(clusters)]
        k = len(groups)
        meds = np.empty(n)
        for i in range(n):
            pick = rng.integers(0, k, size=k)
            meds[i] = np.median(np.concatenate([groups[j] for j in pick]))
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


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
