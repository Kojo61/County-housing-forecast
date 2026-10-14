"""Household formation (headship) and tenure.

For householder age group g (15-24, 25-34, ..., 85+):

    headship rate   HR_g  = householders_g / population_g
    ownership rate  OWN_g = owner householders_g / householders_g

Projected households, owners and renters:

    HH_g    = HR_g * P_g(t)
    Owner_g = HH_g * OWN_g
    Renter_g = HH_g - Owner_g

Rates estimated from the ACS (sampling error) are shrunk toward the state
rate by empirical Bayes using the published margins of error; rates from the
decennial census are used as published. Version 0.1 uses total population in
the denominator (group-quarters population is not removed); see the
methodology's limitations section.
"""

import numpy as np

from .ages import HH_LABELS, N_HH, to_hh_groups
from .shrinkage import empirical_bayes, group_ratio


def _arrays(hb, geoids, column):
    idx = {g: i for i, g in enumerate(geoids)}
    out = np.full((len(geoids), N_HH), np.nan)
    if column not in hb.columns:
        return out
    ages = {a: j for j, a in enumerate(HH_LABELS)}
    for g, a, v in zip(hb["geoid"], hb["hh_age"], hb[column]):
        if g in idx and a in ages:
            out[idx[g], ages[a]] = v
    return out


def _prop_se(p, n, se_x, se_n):
    """ACS standard error of a proportion p = x/n (falls back to the ratio formula)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        rad = se_x ** 2 - (p ** 2) * se_n ** 2
        rad = np.where(rad < 0, se_x ** 2 + (p ** 2) * se_n ** 2, rad)
        return np.where(n > 0, np.sqrt(rad) / n, np.nan)


def rates(hb, geoids, states, shrink=True):
    """Headship and ownership rates (C, 9) from one households_base series.

    hb columns: geoid, hh_age, pop, owners, renters, optional pop_se, owners_se, renters_se.
    Returns dict with hr, own (shrunk if requested), raw versions, weights, and counts.
    """
    pop = _arrays(hb, geoids, "pop")
    owners = _arrays(hb, geoids, "owners")
    renters = _arrays(hb, geoids, "renters")
    hh = owners + renters
    with np.errstate(divide="ignore", invalid="ignore"):
        hr_raw = np.where(pop > 0, hh / pop, np.nan)
        own_raw = np.where(hh > 0, owners / hh, np.nan)
    hr_raw = np.clip(hr_raw, 0, 1)
    own_raw = np.clip(own_raw, 0, 1)
    hr_target = group_ratio(np.where(np.isfinite(hr_raw), hh, 0), np.where(np.isfinite(hr_raw), pop, 0), states)
    own_target = group_ratio(np.where(np.isfinite(own_raw), owners, 0), np.where(np.isfinite(own_raw), hh, 0), states)

    has_se = all(c in hb.columns for c in ("pop_se", "owners_se", "renters_se"))
    w_hr = np.ones_like(hr_raw)
    w_own = np.ones_like(own_raw)
    hr, own = hr_raw.copy(), own_raw.copy()
    if has_se:
        se_pop = np.nan_to_num(_arrays(hb, geoids, "pop_se"))
        se_o = np.nan_to_num(_arrays(hb, geoids, "owners_se"))
        se_r = np.nan_to_num(_arrays(hb, geoids, "renters_se"))
        se_hh = np.sqrt(se_o ** 2 + se_r ** 2)
        se_hr = _prop_se(hr_raw, pop, se_hh, se_pop)
        se_own = _prop_se(own_raw, hh, se_o, se_hh)
        if shrink:
            hr, w_hr, _ = empirical_bayes(hr_raw, se_hr, hr_target)
            own, w_own, _ = empirical_bayes(own_raw, se_own, own_target)
    # Counties with no usable estimate fall back to the state rate.
    hr = np.where(np.isfinite(hr), hr, hr_target)
    own = np.where(np.isfinite(own), own, own_target)
    return {"hr": hr, "own": own, "hr_raw": hr_raw, "own_raw": own_raw,
            "w_hr": w_hr, "w_own": w_own, "pop": pop, "households": hh,
            "owners": owners, "renters": renters}


def scenario_rates(base_hr, alternative_hrs):
    """Low/base/high headship: min/base/max by age across the base and alternatives."""
    stack = np.stack([base_hr] + list(alternative_hrs))
    return {"base": base_hr,
            "low": np.nanmin(stack, axis=0),
            "high": np.nanmax(stack, axis=0)}


def project(pop, hr, own):
    """pop (C, 2, 18) -> dict of households, owners, renters by age group (C, 9)."""
    pg = to_hh_groups(pop.sum(axis=1))
    hh = pg * hr
    owners = hh * own
    return {"households": hh, "owners": owners, "renters": hh - owners}
