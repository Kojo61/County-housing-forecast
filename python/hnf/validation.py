"""Forecast error metrics, empirical prediction intervals and coverage.

For county i with forecast F_i and actual A_i:

    error            e_i  = F_i - A_i
    percent error    pe_i = 100 * (F_i - A_i) / A_i
    MAE    = mean |e|                 RMSE  = sqrt(mean e^2)
    MAPE   = mean |pe|                MedAPE = median |pe|
    MALPE  = mean pe   (bias: positive = over-projection)
    Share within 5% = share of counties with |pe| <= 5

Empirical prediction intervals (EPI): the q_lo and q_hi quantiles of
percent errors from an earlier back-test of the same horizon, by launch-size
class, are applied to new forecasts:

    A = F / (1 + pe/100)  =>  interval [F / (1 + q_hi/100), F / (1 + q_lo/100)]

Coverage = share of actuals that fall inside their interval.
"""

import numpy as np
import pandas as pd


def classify(values, breaks, labels):
    v = np.asarray(values, float)
    idx = np.searchsorted(np.asarray(breaks, float), v, side="right")
    lab = np.asarray(labels, dtype=object)[np.clip(idx, 0, len(labels) - 1)]
    return np.where(np.isfinite(v), lab, "unknown")


def error_frame(geoids, forecast, actual, **cols):
    f = np.asarray(forecast, float)
    a = np.asarray(actual, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pe = np.where(a != 0, 100.0 * (f - a) / a, np.nan)
    df = pd.DataFrame({"geoid": geoids, "forecast": f, "actual": a,
                       "error": f - a, "pct_error": pe, "abs_pct_error": np.abs(pe)})
    for k, v in cols.items():
        df[k] = v
    return df


def summarize(errors, by):
    """Metrics for each group in `by` (list of column names)."""
    e = errors.dropna(subset=["forecast", "actual", "pct_error"])
    rows = []
    for keys, g in e.groupby(by, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(by, keys)),
                     "n": len(g),
                     "MAE": g["error"].abs().mean(),
                     "RMSE": float(np.sqrt((g["error"] ** 2).mean())),
                     "MAPE": g["abs_pct_error"].mean(),
                     "MedAPE": g["abs_pct_error"].median(),
                     "MALPE": g["pct_error"].mean(),
                     "share_within_5pct": (g["abs_pct_error"] <= 5).mean()})
    return pd.DataFrame(rows)


def win_rate(errors, model="model", key=("geoid", "measure")):
    """Share of counties where the model's absolute error beats each benchmark."""
    e = errors.dropna(subset=["abs_pct_error"])
    wide = e.pivot_table(index=list(key), columns="method", values="abs_pct_error")
    rows = []
    if model not in wide.columns:
        return pd.DataFrame(columns=["benchmark", "n", "model_wins_share"])
    for m in wide.columns:
        if m == model:
            continue
        both = wide[[model, m]].dropna()
        if len(both):
            rows.append({"benchmark": m, "n": len(both),
                         "model_wins_share": float((both[model] < both[m]).mean())})
    return pd.DataFrame(rows, columns=["benchmark", "n", "model_wins_share"])


def epi_table(errors, q_lo, q_hi, by="size_class"):
    """Percent-error quantiles by class (plus an 'overall' row) from a calibration back-test."""
    e = errors.dropna(subset=["pct_error"])
    rows = [{by: "overall", "q_lo": e["pct_error"].quantile(q_lo), "q_hi": e["pct_error"].quantile(q_hi), "n": len(e)}]
    for k, g in e.groupby(by):
        rows.append({by: k, "q_lo": g["pct_error"].quantile(q_lo), "q_hi": g["pct_error"].quantile(q_hi), "n": len(g)})
    return pd.DataFrame(rows)


def apply_epi(forecast, classes, table, by="size_class", min_n=20):
    """Interval bounds for each forecast; classes with fewer than min_n errors use 'overall'."""
    t = table.set_index(by)
    lo = np.empty(len(forecast))
    hi = np.empty(len(forecast))
    for i, (f, c) in enumerate(zip(forecast, classes)):
        row = t.loc[c] if (c in t.index and t.loc[c, "n"] >= min_n) else t.loc["overall"]
        lo[i] = f / (1 + row["q_hi"] / 100.0)
        hi[i] = f / (1 + row["q_lo"] / 100.0)
    return lo, hi


def coverage(actual, lo, hi):
    a = np.asarray(actual, float)
    ok = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    return float(((a[ok] >= lo[ok]) & (a[ok] <= hi[ok])).mean()) if ok.any() else np.nan
