"""Shrinkage estimators for small-county rates.

Two estimators are used:

credibility(...)  w = n / (n + K): weight on the county's own rate grows with
                  its exposure n (people, births or permits). K is the
                  exposure at which county and group get equal weight.

empirical_bayes(...)  w = tau^2 / (tau^2 + se^2): for survey estimates with
                  published standard errors (ACS). tau^2, the true between-county
                  variance around the group target, is estimated by the method
                  of moments, pooling counties nationally.
"""

import numpy as np


def group_ratio(numerator, exposure, groups):
    """Pooled group rate sum(numerator)/sum(exposure), broadcast back to members.

    numerator, exposure: arrays whose first axis is counties; groups: (C,) labels.
    """
    num = np.nan_to_num(np.asarray(numerator, float))
    den = np.nan_to_num(np.asarray(exposure, float))
    out = np.full(num.shape, np.nan)
    for g in np.unique(groups):
        idx = groups == g
        n = num[idx].sum(axis=0)
        d = den[idx].sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[idx] = np.where(d > 0, n / d, np.nan)
    return out


def credibility(rate, exposure, group_rate, K):
    """Return (shrunk rate, weight). Cells with zero exposure take the group rate."""
    rate = np.asarray(rate, float)
    n = np.clip(np.nan_to_num(np.asarray(exposure, float)), 0, None)
    w = n / (n + K) if K > 0 else (n > 0).astype(float)
    shrunk = np.where(np.isfinite(rate), w * np.nan_to_num(rate) + (1 - w) * group_rate, group_rate)
    return shrunk, w


def empirical_bayes(estimate, se, target):
    """Shrink survey estimates toward a target. Arrays are (C, k): k separate cells.

    tau^2 for each cell column is max(0, var(estimate - target) - mean(se^2)).
    Estimates with se == 0 (full-count census data) keep weight 1.
    """
    est = np.asarray(estimate, float)
    se2 = np.asarray(se, float) ** 2
    tgt = np.asarray(target, float)
    dev = est - tgt
    ok = np.isfinite(dev) & np.isfinite(se2)
    k = est.shape[1]
    tau2 = np.zeros(k)
    for j in range(k):
        m = ok[:, j]
        if m.sum() > 2:
            tau2[j] = max(0.0, np.var(dev[m, j]) - np.mean(se2[m, j]))
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(se2 > 0, tau2 / (tau2 + se2), 1.0)
    w = np.where(np.isfinite(w), w, 0.0)
    shrunk = np.where(np.isfinite(est), w * np.nan_to_num(est) + (1 - w) * tgt, tgt)
    return shrunk, w, tau2
