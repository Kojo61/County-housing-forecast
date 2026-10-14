"""Optional employment-informed migration level.

Estimated across counties for a past period, with state fixed effects:

    mig_c = a + b1 * emp_growth_c + b2 * wage_growth_c + b3 * prior_mig_c + state_s + e_c

mig is the five-year crude net migration rate of the target window; growth
rates are annualized log changes in QCEW total covered employment and average
weekly wage over the years *before* that window; prior_mig is the crude rate
of the preceding window. The fitted equation is then applied at launch, using
the employment years just before launch, to predict the next five-year crude
rate. The base migration schedule is shifted to that level
(migration.shift_schedule).

The add-on is kept only if back-test B5 shows it lowers out-of-sample error
(config: employment.use_in_production).
"""

import numpy as np
import pandas as pd


def annual_log_growth(emp, geoids, y0, y1, column):
    """Annualized log growth of `column` between years y0 and y1 for each county."""
    e = emp[emp["year"].isin([y0, y1])].pivot_table(index="geoid", columns="year", values=column)
    e = e.reindex(geoids)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(e[y1].to_numpy(float) / e[y0].to_numpy(float)) / (y1 - y0)
    return np.where(np.isfinite(g), g, np.nan)


def design(features, states, state_levels):
    """Design matrix: intercept, features, state dummies (first state is the reference)."""
    cols = [np.ones(len(states))]
    names = ["intercept"]
    for name, x in features.items():
        cols.append(np.asarray(x, float))
        names.append(name)
    for s in state_levels[1:]:
        cols.append((np.asarray(states) == s).astype(float))
        names.append(f"state_{s}")
    return np.column_stack(cols), names


def fit(y, X):
    """OLS with heteroskedasticity-robust (HC1) standard errors."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[ok], X[ok]
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = (X * resid[:, None] ** 2).T @ X
    cov = XtX_inv @ meat @ XtX_inv * (n / max(n - k, 1))
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / tss if tss > 0 else np.nan
    return {"beta": beta, "se": np.sqrt(np.clip(np.diag(cov), 0, None)), "r2": r2, "n": n}


def fit_and_predict(train_y, train_feats, apply_feats, states):
    """Fit on the training period and predict the next-period crude rate for every county."""
    levels = sorted(pd.unique(np.asarray(states)))
    X_train, names = design(train_feats, states, levels)
    model = fit(np.asarray(train_y, float), X_train)
    X_apply, _ = design(apply_feats, states, levels)
    pred = X_apply @ model["beta"]
    table = pd.DataFrame({"term": names, "estimate": model["beta"], "std_error": model["se"]})
    table["r2"] = model["r2"]
    table["n"] = model["n"]
    return pred, table
