"""Households by HUD income band (HAMFI), using CHAS shares.

    HH[tenure, band](t) = HH[tenure](t) * share[tenure, band]

Shares are each county's CHAS distribution within tenure, held constant in
Version 0.1. Bands: <=30%, 30-50%, 50-80%, 80-100%, >100% of HUD Area Median
Family Income.
"""

import numpy as np

TENURES = ["owner", "renter"]


def shares(chas, geoids, bands):
    """CHAS household counts -> shares (C, 2, n_bands); NaN where a county is missing."""
    idx = {g: i for i, g in enumerate(geoids)}
    t_idx = {t: j for j, t in enumerate(TENURES)}
    b_idx = {b: j for j, b in enumerate(bands)}
    counts = np.full((len(geoids), 2, len(bands)), np.nan)
    for g, t, b, v in zip(chas["geoid"], chas["tenure"], chas["band"], chas["households"]):
        if g in idx and t in t_idx and b in b_idx:
            counts[idx[g], t_idx[t], b_idx[b]] = v
    total = counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(total > 0, counts / total, np.nan)


def by_band(owners, renters, share):
    """owners, renters: (C,) totals -> (C, 2, n_bands) households by tenure and band."""
    return np.stack([owners, renters], axis=1)[:, :, None] * share
