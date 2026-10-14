"""Parser tests: raw files laid out like the real downloads -> processed tables."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE))
from fixtures import write_raw  # noqa: E402
from hnf import prepare  # noqa: E402
from hnf.io import load_config, read_table  # noqa: E402


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("prep")
    raw, out = root / "raw", root / "processed"
    truth = write_raw(raw)
    cfg = load_config()
    prepare.CHECKS.clear()
    failed = prepare.prepare_all(cfg, raw, out)
    return out, truth, failed


def test_all_checks_pass(processed):
    out, _, failed = processed
    assert failed == [], failed


def test_population_series_and_codes(processed):
    out, truth, _ = processed
    pop = read_table(out, "population")
    assert set(pop["series"]) >= {"v2025", "int2020", "int2000", "census2010", "census2020"}
    v = pop[(pop.series == "v2025")]
    assert sorted(v["date"].unique()) == ["2020-04-01"] + [f"{y}-07-01" for y in range(2020, 2026)]
    assert v["age"].between(0, 17).all() and set(v["sex"]) == {"M", "F"}
    # Census 2010 from SF1 cells equals the fixture's April 2010 population by age and sex.
    c = pop[(pop.series == "census2010") & (pop.geoid == "55025")].sort_values(["sex", "age"])
    expected = np.concatenate([truth["census2010"]["025"][1], truth["census2010"]["025"][0]])  # F then M
    assert np.allclose(c["pop"].to_numpy(), expected)


def test_births_drop_partial_year(processed):
    out, _, _ = processed
    b = read_table(out, "births")
    assert sorted(b["year_end"].unique()) == [2021, 2022, 2023, 2024, 2025]


def test_households_and_acs_errors(processed):
    out, _, _ = processed
    hb = read_table(out, "households_base")
    assert set(hb["series"]) == {"sf1_2010", "dhc_2020", "acs5_2024"}
    acs = hb[hb.series == "acs5_2024"]
    assert acs["owners_se"].notna().all() and (acs["owners_se"] > 0).all()
    assert len(acs[acs.geoid == "55025"]) == 9


def test_housing_units_permits_chas_life_tables_employment(processed):
    out, _, _ = processed
    hu = read_table(out, "housing_units")
    dane = hu[(hu.series == "v2025") & (hu.geoid == "55025")].set_index("date")["hu"]
    assert dane["2025-07-01"] == 255000 and dane["2020-04-01"] == 240000
    permits = read_table(out, "permits")
    assert (permits["units"] == 10 + 2 + 4 + 40).all()          # estimates-with-imputation columns
    chas = read_table(out, "chas")
    assert chas.groupby(["geoid", "tenure"])["households"].sum().loc[("55025", "owner")] == 580
    lt = read_table(out, "life_tables")
    assert len(lt) == 2 * 101 and lt["age"].max() == 100
    emp = read_table(out, "employment")
    assert set(emp["year"]) == {2019, 2025} and (emp["emp"] > 5000).all()
