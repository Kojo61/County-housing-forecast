"""End-to-end runs on the synthetic dataset (python/make_synthetic.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
import make_synthetic  # noqa: E402
from hnf import pipeline  # noqa: E402
from hnf.io import load_config  # noqa: E402


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    make_synthetic.main(d, n_per_state=8, seed=7)
    return d


def test_production_outputs_are_complete_and_ordered(synthetic):
    res = pipeline.run("production", load_config(), synthetic)
    s = res["summary"]
    assert len(res["skipped"]) == 0
    assert not s["value"].isna().any()
    final = s[(s["measure"] == "gap") & (s["horizon"] == 10)].pivot_table(
        index="geoid", columns="scenario", values="value")
    assert ((final["low"] <= final["base"]) & (final["base"] <= final["high"])).all()
    w = s[s["measure"].isin(["gap", "existing_gap", "growth_in_need", "pipeline"])].pivot_table(
        index=["geoid", "scenario", "horizon"], columns="measure", values="value")
    assert np.allclose(w["gap"], w["existing_gap"] + w["growth_in_need"] - w["pipeline"])
    # Every scenario starts from the same households at launch.
    hh0 = s[(s["measure"] == "households") & (s["horizon"] == 0)].pivot_table(
        index="geoid", columns="scenario", values="value")
    assert np.allclose(hh0["low"], hh0["high"])
    # Income bands add up to tenure totals.
    base5 = s[(s["scenario"] == "base") & (s["horizon"] == 5)].pivot_table(
        index="geoid", columns="measure", values="value")
    bands = ["le30", "30_50", "50_80", "80_100", "gt100"]
    assert np.allclose(base5[[f"owner_{b}" for b in bands]].sum(axis=1), base5["owners"])


def test_backtests_score_without_leakage(synthetic):
    cfg = load_config()
    for name in ("testA10", "testA5", "testB5"):
        res = pipeline.run(name, cfg, synthetic)
        e = res["errors"]
        assert set(e["method"]) >= {"model", "CONST", "LIN", "EXP", "SHR", "AVE"}
        assert e.loc[e["method"] == "model", "abs_pct_error"].median() < 15
    # The information-set rule: every calibration input ends at or before launch.
    for name in ("testA10", "testA5", "testB5"):
        spec = cfg["runs"][name]
        launch = spec["launch"]["date"]
        launch_year = int(launch[:4])
        for w in spec["migration_windows"]:
            assert w["end"]["date"] <= launch, (name, w["id"])
        assert spec["fertility"]["index"]["years"][1] <= launch_year
        assert spec["fertility"]["asfr_year"] < launch_year
        if "supply" in spec:
            assert spec["supply"]["realization"]["permit_years"][1] < launch_year
            assert spec["supply"]["projection_permit_years"][1] < launch_year
