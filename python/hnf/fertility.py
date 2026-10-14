"""County fertility index and projected births.

Births in a five-year step use national age-specific fertility rates (ASFR)
scaled by a county fertility index k (indirect standardization):

    expected births E_c = sum over years y and ages a of ASFR_a * W_c,a,y
    k_c = (B_c / E_c) / (B_US / E_US)

where B_c are county births observed in the calibration window (PEP
components of change) and W are women by age at the start of each birth year.
Dividing by the national ratio makes k a *relative* index, so it does not
depend on which year's national schedule is used as the reference. Small
counties are shrunk toward their state's index with weight E_c / (E_c + K).

Projected births for a step from t to t+5:

    B = 5 * k_c * sum_a ASFR_a * (W_a(t) + W_a(t+5)) / 2
"""

import numpy as np

from .ages import FERT_IDX
from .shrinkage import credibility, group_ratio


def expected_births(women_by_year, asfr):
    """women_by_year: (Y, C, 8) women aged 10-14..45-49; asfr: (8,) per woman per year."""
    return (np.asarray(women_by_year) * np.asarray(asfr)).sum(axis=(0, 2))


def fertility_index(births, expected, states, K):
    """Return (k shrunk, k raw, weight). births, expected: (C,) totals over the window."""
    births = np.asarray(births, float)
    expected = np.asarray(expected, float)
    ok = np.isfinite(births) & np.isfinite(expected) & (expected > 0)
    national = births[ok].sum() / expected[ok].sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        k_raw = np.where(ok, births / expected / national, np.nan)
    k_state = group_ratio(np.where(ok, births, 0.0), np.where(ok, expected, 0.0), states) / national
    exposure = np.where(ok, expected, 0.0)
    k, w = credibility(k_raw, exposure, k_state, K)
    k = np.where(np.isfinite(k), k, 1.0)
    return k, k_raw, w


def step_births(P_t, P_t5, asfr, k):
    """Births during a five-year step: (C,). P arrays are (C, 2, 18)."""
    women = 0.5 * (P_t[:, 1, FERT_IDX] + P_t5[:, 1, FERT_IDX])
    return 5.0 * k * (women * asfr).sum(axis=1)
