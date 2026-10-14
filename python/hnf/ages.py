"""Age-group conventions used everywhere in the model.

Population is held in 18 five-year groups indexed 0..17:
0 = 0-4, 1 = 5-9, ..., 16 = 80-84, 17 = 85+ (open-ended).
Sex is indexed 0 = male, 1 = female (labels "M" and "F").
Arrays of population are shaped (counties, 2, 18).
"""

import numpy as np

N_AGE = 18
N_SEX = 2
SEXES = ["M", "F"]
AGE_LABELS = [f"{5 * i}-{5 * i + 4}" for i in range(17)] + ["85+"]

# Female ages that carry fertility: 10-14 .. 45-49 (age indices 2..9).
FERT_IDX = np.arange(2, 10)

# Householder age groups used by the tenure-by-age tables
# (ACS B25007, 2010 SF1 H17, 2020 DHC H13) mapped to five-year age indices.
HH_GROUPS = {
    "15-24": [3, 4],
    "25-34": [5, 6],
    "35-44": [7, 8],
    "45-54": [9, 10],
    "55-59": [11],
    "60-64": [12],
    "65-74": [13, 14],
    "75-84": [15, 16],
    "85+": [17],
}
HH_LABELS = list(HH_GROUPS)
N_HH = len(HH_LABELS)

# Sex-by-age cells shared by ACS B01001, 2010 SF1 P12 and 2020 DHC P12.
# Male cells are 3..25 and female cells 27..49 (female = male + 24).
SEX_BY_AGE_CELLS = {
    0: [3], 1: [4], 2: [5], 3: [6, 7], 4: [8, 9, 10], 5: [11], 6: [12],
    7: [13], 8: [14], 9: [15], 10: [16], 11: [17], 12: [18, 19],
    13: [20, 21], 14: [22], 15: [23], 16: [24], 17: [25],
}
FEMALE_OFFSET = 24


def to_hh_groups(pop_both_sexes):
    """Collapse a (..., 18) array of population into the 9 householder age groups."""
    return np.stack(
        [pop_both_sexes[..., idx].sum(axis=-1) for idx in HH_GROUPS.values()], axis=-1
    )
