"""Naive benchmark forecasts the model must beat.

Inputs are county totals at a long base date (about 10 years before launch),
a short base date (about 5 years before launch) and the launch date.

    CONST       P(t+h) = P(t)
    LIN         P(t+h) = P(t) + h * [P(t) - P(t-L)] / L
    EXP         P(t+h) = P(t) * [P(t) / P(t-L)] ** (h / L)        (compound growth, CAGR)
    EXP_RECENT  same as EXP using the short base period
    SHR         P(t+h) = [P(t) / State(t)] * State_LIN(t+h)      (constant share of state)
    AVE         mean of LIN, EXP and SHR (often hard to beat in the literature)
"""

import numpy as np


def _group_sum(x, groups):
    out = np.empty_like(x, dtype=float)
    for g in np.unique(groups):
        idx = groups == g
        out[idx] = np.nansum(x[idx])
    return out


def population_benchmarks(p_long, p_short, p_launch, years_long, years_short, horizon, states):
    p_long, p_short, p_launch = (np.asarray(v, float) for v in (p_long, p_short, p_launch))
    with np.errstate(divide="ignore", invalid="ignore"):
        lin = np.maximum(p_launch + horizon * (p_launch - p_long) / years_long, 0.0)
        exp_ = p_launch * (p_launch / p_long) ** (horizon / years_long)
        exp_recent = p_launch * (p_launch / p_short) ** (horizon / years_short)
        st_launch = _group_sum(p_launch, states)
        st_long = _group_sum(p_long, states)
        st_proj = st_launch + horizon * (st_launch - st_long) / years_long
        shr = p_launch / st_launch * st_proj
    out = {"CONST": p_launch.copy(), "LIN": lin, "EXP": exp_, "EXP_RECENT": exp_recent, "SHR": shr}
    out["AVE"] = (lin + exp_ + shr) / 3.0
    return out


def linear_trend(v_start, v_launch, years_base, horizon):
    """Linear extrapolation of any series (used for households and housing units)."""
    v_start, v_launch = np.asarray(v_start, float), np.asarray(v_launch, float)
    return np.maximum(v_launch + horizon * (v_launch - v_start) / years_base, 0.0)
