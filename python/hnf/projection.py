"""Cohort-component projection engine (five-year steps, all counties at once).

One step from t to t+5, for each county and sex:

    P[a+1](t+5) = P[a](t) * (S[a] + m[a])                         a = 0..15
    P[85+](t+5) = P[80-84](t) * (S[16] + m[16]) + P[85+](t) * (S[17] + m[17])
    B           = 5 * k * sum_a ASFR_a * (W_a(t) + W_a(t+5)) / 2  (women 10-49)
    P[0-4](t+5) = B * share_sex * (Sb + mb)

with share_sex = SRB/(1+SRB) for males and 1/(1+SRB) for females. The survival
plus migration factor is floored (config: min_survival_plus_migration) so that
no cell can turn negative.
"""

import numpy as np

from .fertility import step_births


def project(P0, S, Sb, m, mb, asfr, k, steps, srb=1.05, floor=0.0):
    """Project P0 (C, 2, 18) for `steps` five-year steps.

    Returns (pops, births): pops is a list of steps+1 arrays (launch first);
    births is a list of (C, 2) arrays, one per step.
    """
    factor = np.maximum(S + m, floor)
    factor_b = np.maximum(Sb + mb, floor)
    male_share = srb / (1.0 + srb)
    sex_share = np.array([male_share, 1.0 - male_share])
    pops, births_out = [np.asarray(P0, float)], []
    P = pops[0]
    for _ in range(steps):
        Pn = np.zeros_like(P)
        Pn[:, :, 1:17] = P[:, :, 0:16] * factor[:, :, 0:16]
        Pn[:, :, 17] = P[:, :, 16] * factor[:, :, 16] + P[:, :, 17] * factor[:, :, 17]
        B = step_births(P, Pn, asfr, k)                 # women 10-49 do not depend on births
        Bs = B[:, None] * sex_share[None, :]
        Pn[:, :, 0] = Bs * factor_b
        pops.append(Pn)
        births_out.append(Bs)
        P = Pn
    return pops, births_out
