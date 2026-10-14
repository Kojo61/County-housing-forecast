"""Five-year survival ratios from single-year period life tables.

Input life tables follow the NCHS layout: one row per single year of age
0, 1, ..., 99 and an open interval "100 and over", with columns qx, lx, dx,
Lx, Tx, ex (radix l0 = 100,000).

For five-year age group a (0 = 0-4, ..., 16 = 80-84, 17 = 85+) the survival
ratio S[a] moves people alive in group a at time t into group a+1 at t+5:

    S[a]  = 5L(5a+5) / 5L(5a)         a = 0..15
    S[16] = 5L85 / 5L80               (80-84 -> 85-89, part of 85+)
    S[17] = T90 / T85                 (85+  -> 90+,   part of 85+)
    Sb    = 5L0 / (5 * l0)            (births in the period -> 0-4 at t+5)
"""

import numpy as np

from .ages import N_AGE


def survival_ratios(Lx, l0):
    """Return (S, Sb) from single-year Lx for ages 0..99 plus the open 100+ value."""
    L = np.asarray(Lx, dtype=float)
    if L.shape[0] != 101:
        raise ValueError(f"Expected 101 Lx values (ages 0..99 and 100+), got {L.shape[0]}")
    T = np.cumsum(L[::-1])[::-1]                       # T_x = sum of L_y for y >= x
    L5 = np.array([L[5 * g: 5 * g + 5].sum() for g in range(N_AGE)])   # 0-4 .. 85-89
    S = np.empty(N_AGE)
    S[:16] = L5[1:17] / L5[0:16]
    S[16] = L5[17] / L5[16]
    S[17] = T[90] / T[85]
    Sb = L5[0] / (5.0 * float(l0))
    return S, Sb


def survival_lookup(life_tables):
    """Build {(table_id, geo, sex): (S, Sb)} from the processed life_tables table."""
    out = {}
    for (table_id, geo, sex), g in life_tables.groupby(["table_id", "geo", "sex"]):
        g = g.sort_values("age")
        out[(table_id, geo, sex)] = survival_ratios(g["Lx"].to_numpy(), g["lx"].iloc[0])
    return out


def county_survival(states, table_id, lookup):
    """Survival arrays for each county: S (C, 2, 18) and Sb (C, 2).

    A state-level table is used when the table set has one for the county's
    state; otherwise the national ("US") table of the same id.
    """
    C = len(states)
    S = np.empty((C, 2, N_AGE))
    Sb = np.empty((C, 2))
    for i, st in enumerate(states):
        for s, sex in enumerate(("M", "F")):
            key = (table_id, st, sex)
            if key not in lookup:
                key = (table_id, "US", sex)
            if key not in lookup:
                raise KeyError(
                    f"Life table '{table_id}' has neither a state ({st}) nor a US table "
                    f"for sex {sex}. Check config/sources.yaml and data/processed/life_tables.csv.")
            S[i, s], Sb[i, s] = lookup[key]
    return S, Sb
