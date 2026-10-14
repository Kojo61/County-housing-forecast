"""Cohort-residual net migration, small-county shrinkage and scenario schedules.

For a calibration window running from t to t+n (n is normally 5 years):

    NM[a+1] = P[a+1](t+n) - P[a](t) * S[a]                      a = 0..15
    NM[85+] = P[85+](t+n) - (P[80-84](t) * S[16] + P[85+](t) * S[17])
    NM[0-4] = P[0-4](t+n) - B * Sb        (only when county births B are known)

Rates are expressed on the start-of-window population of the cohort, so a
projection step is P[a+1](t+5) = P[a](t) * (S[a] + m[a]). Windows that are
not exactly five years long are rescaled to a five-year rate (m * 5 / n).

When births for the window are not available, the rate for children born
during a step is beta times the 0-4 -> 5-9 cohort rate (they are exposed for
about half the period); beta is set in config/model.yaml.
"""

import numpy as np

from .shrinkage import credibility, group_ratio


def _div(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, np.nan)


def residual(P_start, P_end, S, Sb, births=None):
    """Net migrants and exposures for one window (no rescaling, no shrinkage).

    P_start, P_end: (C, 2, 18). S: (C, 2, 18). Sb: (C, 2). births: (C, 2) or None.
    Returns dict with NM (C,2,18) by from-age (index 16 and 17 both hold the 85+
    total), exposure (C,2,18), NM_b and exposure_b (C,2) for births (NaN if unknown).
    """
    C = P_start.shape[0]
    NM = np.zeros((C, 2, 18))
    NM[:, :, 0:16] = P_end[:, :, 1:17] - P_start[:, :, 0:16] * S[:, :, 0:16]
    survivors_85 = P_start[:, :, 16] * S[:, :, 16] + P_start[:, :, 17] * S[:, :, 17]
    nm85 = P_end[:, :, 17] - survivors_85
    exposure = P_start.astype(float).copy()
    e85 = P_start[:, :, 16] + P_start[:, :, 17]
    NM[:, :, 16] = nm85
    NM[:, :, 17] = nm85
    exposure[:, :, 16] = e85
    exposure[:, :, 17] = e85
    if births is not None:
        NM_b = P_end[:, :, 0] - births * Sb
        exposure_b = births.astype(float)
    else:
        NM_b = np.full((C, 2), np.nan)
        exposure_b = np.full((C, 2), np.nan)
    return {"NM": NM, "exposure": exposure, "NM_b": NM_b, "exposure_b": exposure_b}


def window_rates(P_start, P_end, S, Sb, years, states, K, beta, births=None):
    """Five-year net migration rates for one window, raw and shrunk toward the state.

    Returns a dict with m_raw, m (shrunk), w (weights), mb_raw, mb, crude
    (net migrants aged 5+ at window end per start population, five-year basis)
    and NM (rescaled net migrants by from-age).
    """
    r = residual(P_start, P_end, S, Sb, births)
    scale = 5.0 / years
    NM = r["NM"] * scale
    exposure = r["exposure"]
    m_raw = _div(NM, exposure)
    m_state = group_ratio(NM, exposure, states)
    m, w = credibility(m_raw, exposure, m_state, K)

    if births is not None:
        NM_b = r["NM_b"] * scale
        mb_raw = _div(NM_b, r["exposure_b"])
        mb_state = group_ratio(NM_b, r["exposure_b"], states)
        mb, _ = credibility(mb_raw, r["exposure_b"], mb_state, K)
    else:
        mb_raw = beta * m_raw[:, :, 0]
        mb = beta * m[:, :, 0]

    # Net migrants aged 5+ at the end of the window (85+ counted once).
    nm_5plus = NM[:, :, 0:17].sum(axis=(1, 2))
    crude = _div(nm_5plus, P_start.sum(axis=(1, 2)))
    return {"m_raw": m_raw, "m": m, "w": w, "m_state": m_state,
            "mb_raw": mb_raw, "mb": mb, "crude": crude, "NM": NM, "years": years}


def implied_net_migrants(P, m):
    """Five-year net migrants aged 5+ implied by schedule m applied to population P."""
    total = (P[:, :, 0:16] * m[:, :, 0:16]).sum(axis=(1, 2))
    total += ((P[:, :, 16] + P[:, :, 17]) * m[:, :, 16]).sum(axis=1)
    return total


def scenario_schedules(windows, weights, P0):
    """Base, low and high migration schedules.

    base = weighted average of the window schedules.
    low / high = for each county, the observed window whose schedule implies the
    fewest / most net migrants when applied to the launch population.
    windows: list of dicts from window_rates (in the same order as weights).
    """
    wts = np.asarray(weights, float)
    wts = wts / wts.sum()
    M = np.stack([w["m"] for w in windows])            # (W, C, 2, 18)
    MB = np.stack([w["mb"] for w in windows])          # (W, C, 2)
    base_m = np.tensordot(wts, M, axes=1)
    base_mb = np.tensordot(wts, MB, axes=1)
    implied = np.stack([implied_net_migrants(P0, w["m"]) for w in windows], axis=1)  # (C, W)
    lo = implied.argmin(axis=1)
    hi = implied.argmax(axis=1)
    c = np.arange(P0.shape[0])
    return {
        "base": (base_m, base_mb),
        "low": (M[lo, c], MB[lo, c]),
        "high": (M[hi, c], MB[hi, c]),
    }


def shift_schedule(m, mb, P0, m_state, delta):
    """Add delta (a change in the five-year crude rate) to a schedule.

    The shift is spread across ages in proportion to the absolute state rates
    (so it lands where migration actually happens) and normalized so that the
    county's crude rate changes by exactly delta.
    """
    profile = np.abs(np.nan_to_num(m_state))
    weight = implied_net_migrants(P0, profile)
    pop = P0.sum(axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(weight > 0, delta * pop / weight, 0.0)
    m_new = m + profile * scale[:, None, None]
    mb_new = mb + profile[:, :, 0] * scale[:, None]
    return m_new, mb_new
