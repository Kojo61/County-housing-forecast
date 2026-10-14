"""Housing-unit requirement and gap.

Requirement (year-round, market units needed to house projected households
with a functioning vacancy allowance), by tenure:

    Required(t) = Owners(t) / (1 - v_own) + Renters(t) / (1 - v_rent)

Available supply:

    Stock(t)     = HU(launch) + additions(t)
    Available(t) = Stock(t) * available share

Gap (positive = shortfall, negative = surplus) and its exact decomposition:

    Gap(t) = Required(t) - Available(t)
           = [Required(launch) - Available(launch)]       existing gap
           + [Required(t) - Required(launch)]             growth in need
           - [additions(t) * available share]             expected pipeline
"""

import numpy as np


def required_units(owners, renters, v_own, v_rent):
    return owners / (1.0 - v_own) + renters / (1.0 - v_rent)


def gap_components(required_t, required_0, hu_0, additions_t, avail_share):
    available_0 = hu_0 * avail_share
    pipeline = additions_t * avail_share
    available_t = available_0 + pipeline
    return {
        "required_units": required_t,
        "stock": hu_0 + additions_t,
        "available_units": available_t,
        "gap": required_t - available_t,
        "existing_gap": required_0 - available_0,
        "growth_in_need": required_t - required_0,
        "pipeline": pipeline,
    }


def check_identity(c, tol=1e-6):
    lhs = c["gap"]
    rhs = c["existing_gap"] + c["growth_in_need"] - c["pipeline"]
    return np.allclose(lhs, rhs, atol=tol, equal_nan=True)
