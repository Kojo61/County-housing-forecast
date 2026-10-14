"""Housing supply: base stock, permit realization factor and expected additions.

Realization factor (net units added per unit authorized), estimated between
two census counts so it does not depend on the permit-based PEP estimates:

    RF_c = (HU_c(end) - HU_c(start)) / sum of permitted units in the window

RF captures completions, demolitions, conversions, manufactured homes (not
permitted) and areas outside permit-issuing places, so it can be above 1.
It is shrunk toward the state factor with weight P / (P + K), P = permitted
units, and bounded (config). Expected net additions over h years:

    additions(h) = RF_c * annual permits * h

where annual permits are the mean of the recent permit years named in the
run specification (low/high supply use the lowest/highest single year).

Available (year-round, market) stock removes units held for seasonal,
recreational or occasional use, for migrant workers, and (by default) the
"other vacant" units, using the county's current shares:

    available share = 1 - (seasonal + migrant + (1 - alpha) * other vacant) / total units
"""

import numpy as np

from .shrinkage import credibility, group_ratio


def realization_factor(hu_start, hu_end, permits, states, K, bounds):
    """Return dict with rf (shrunk, bounded), rf_raw, rf_state and weight. Inputs (C,)."""
    hu_start = np.asarray(hu_start, float)
    hu_end = np.asarray(hu_end, float)
    permits = np.asarray(permits, float)
    change = hu_end - hu_start
    ok = np.isfinite(change) & np.isfinite(permits)
    with np.errstate(divide="ignore", invalid="ignore"):
        rf_raw = np.where(ok & (permits > 0), change / permits, np.nan)
    rf_state = group_ratio(np.where(ok, change, 0.0), np.where(ok, permits, 0.0), states)
    rf, w = credibility(rf_raw, np.where(ok, permits, 0.0), rf_state, K)
    rf = np.clip(rf, bounds[0], bounds[1])
    return {"rf": rf, "rf_raw": rf_raw, "rf_state": rf_state, "w": w}


def available_share(vac, other_available=0.0):
    """vac: dict of (C,) arrays hu_total, seasonal, migrant, other_vacant."""
    unavailable = vac["seasonal"] + vac["migrant"] + (1.0 - other_available) * vac["other_vacant"]
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(vac["hu_total"] > 0, 1.0 - unavailable / vac["hu_total"], np.nan)
    return np.clip(share, 0.0, 1.0)


def expected_additions(rf, annual_permits, years):
    return rf * annual_permits * years
