"""Unit tests of the demographic and housing arithmetic (Python engine)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from hnf import benchmarks, fertility, gap, households, migration, mortality, projection, shrinkage, supply, validation  # noqa: E402
from hnf.ages import HH_LABELS  # noqa: E402


def constant_hazard_table(mu):
    """Life table with a constant force of mortality mu (closed-form survival ratios)."""
    x = np.arange(101)
    l = 100000.0 * np.exp(-mu * x)
    L = l * (1 - np.exp(-mu)) / mu
    L[100] = l[100] / mu
    return L, l[0]


def test_survival_constant_hazard_is_exact():
    mu = 0.02
    L, l0 = constant_hazard_table(mu)
    S, Sb = mortality.survival_ratios(L, l0)
    assert np.allclose(S, np.exp(-5 * mu))
    assert Sb == pytest.approx((1 - np.exp(-5 * mu)) / (5 * mu))


def random_population(rng, C=7):
    return rng.uniform(100, 5000, size=(C, 2, 18))


def test_residual_recovers_known_rates():
    """Projecting with known rates and re-estimating them must return the same rates."""
    rng = np.random.default_rng(1)
    C = 7
    P0 = random_population(rng, C)
    S = np.clip(rng.uniform(0.8, 0.999, (C, 2, 18)), 0, 1)
    Sb = rng.uniform(0.97, 0.995, (C, 2))
    m = rng.normal(0, 0.05, (C, 2, 18))
    m[:, :, 17] = m[:, :, 16]                       # 85+ shares one rate
    mb = rng.normal(0, 0.03, (C, 2))
    asfr = np.array([0.0002, 0.013, 0.056, 0.09, 0.094, 0.054, 0.013, 0.001])
    k = np.ones(C)
    pops, births = projection.project(P0, S, Sb, m, mb, asfr, k, steps=1)
    est = migration.window_rates(P0, pops[1], S, Sb, 5.0, np.array(["55"] * C), K=0, beta=0.5,
                                 births=births[0])
    assert np.allclose(est["m_raw"], m)
    assert np.allclose(est["mb_raw"], mb)


def test_projection_without_migration_or_births_only_ages_and_dies():
    rng = np.random.default_rng(2)
    P0 = random_population(rng, 3)
    S = np.full((3, 2, 18), 0.98)
    Sb = np.full((3, 2), 0.99)
    zeros = np.zeros((3, 2, 18))
    pops, births = projection.project(P0, S, Sb, zeros, np.zeros((3, 2)), np.zeros(8), np.ones(3), steps=1)
    P1 = pops[1]
    assert np.allclose(P1[:, :, 1:17], P0[:, :, 0:16] * 0.98)
    assert np.allclose(P1[:, :, 17], (P0[:, :, 16] + P0[:, :, 17]) * 0.98)
    assert np.allclose(P1[:, :, 0], 0)


def test_migration_floor_prevents_negative_population():
    P0 = np.full((1, 2, 18), 1000.0)
    S = np.full((1, 2, 18), 0.95)
    m = np.full((1, 2, 18), -2.0)                   # absurd out-migration
    pops, _ = projection.project(P0, S, np.full((1, 2), 0.99), m, np.full((1, 2), -2.0),
                                 np.zeros(8), np.ones(1), steps=2)
    assert (pops[-1] >= 0).all()


def test_shrinkage_weights():
    shrunk, w = shrinkage.credibility(np.array([0.10, 0.10]), np.array([1000.0, 0.0]), np.array([0.0, 0.0]), K=1000)
    assert w[0] == pytest.approx(0.5) and shrunk[0] == pytest.approx(0.05)
    assert shrunk[1] == pytest.approx(0.0)           # no exposure -> group rate
    est = np.array([[0.4], [0.5], [0.6]])
    sh, w, _ = shrinkage.empirical_bayes(est, np.zeros_like(est), np.full_like(est, 0.5))
    assert np.allclose(sh, est) and np.allclose(w, 1.0)   # census data (se = 0) are not shrunk


def test_fertility_index_is_relative_to_nation():
    births = np.array([200.0, 100.0, 300.0])
    expected = np.array([100.0, 100.0, 100.0])      # national ratio = 2
    k, k_raw, _ = fertility.fertility_index(births, expected, np.array(["01", "01", "02"]), K=0)
    assert np.allclose(k_raw, [1.0, 0.5, 1.5])


def test_household_projection_and_tenure():
    pop = np.zeros((1, 2, 18))
    pop[0, :, 5] = 500                              # 1,000 people aged 25-29
    hr = np.zeros((1, 9)); hr[0, 1] = 0.4
    own = np.full((1, 9), 0.25)
    h = households.project(pop, hr, own)
    assert h["households"].sum() == pytest.approx(400)
    assert h["owners"].sum() == pytest.approx(100)
    assert h["renters"].sum() == pytest.approx(300)
    assert HH_LABELS[1] == "25-34"


def test_required_units_and_gap_identity():
    req0 = gap.required_units(np.array([600.0]), np.array([400.0]), 0.02, 0.06)
    assert req0[0] == pytest.approx(600 / 0.98 + 400 / 0.94)
    req5 = gap.required_units(np.array([700.0]), np.array([450.0]), 0.02, 0.06)
    comp = gap.gap_components(req5, req0, np.array([1100.0]), np.array([60.0]), np.array([0.9]))
    assert gap.check_identity(comp)
    assert comp["gap"][0] == pytest.approx(req5[0] - (1100 + 60) * 0.9)


def test_realization_factor_and_available_share():
    rf = supply.realization_factor(np.array([1000.0, 500.0]), np.array([1800.0, 500.0]),
                                   np.array([1000.0, 0.0]), np.array(["01", "01"]), K=0, bounds=[-1, 3])
    assert rf["rf"][0] == pytest.approx(0.8)
    assert rf["rf"][1] == pytest.approx(0.8)        # no permits -> state factor
    share = supply.available_share({"hu_total": np.array([100.0]), "seasonal": np.array([20.0]),
                                    "migrant": np.array([0.0]), "other_vacant": np.array([5.0])})
    assert share[0] == pytest.approx(0.75)


def test_benchmarks_and_metrics():
    b = benchmarks.population_benchmarks(np.array([100.0]), np.array([110.0]), np.array([121.0]),
                                         10, 5, 10, np.array(["01"]))
    assert b["LIN"][0] == pytest.approx(142.0)
    assert b["EXP"][0] == pytest.approx(121 * 1.21)
    e = validation.error_frame(["a", "b"], [110, 90], [100, 100], method="model", measure="pop")
    s = validation.summarize(e, ["method"])
    assert s["MAPE"].iloc[0] == pytest.approx(10.0)
    assert s["MALPE"].iloc[0] == pytest.approx(0.0)
    assert s["RMSE"].iloc[0] == pytest.approx(10.0)


def test_empirical_interval_coverage():
    rng = np.random.default_rng(3)
    actual = rng.uniform(1000, 5000, 400)
    pe = rng.normal(0, 4, 400)
    forecast = actual * (1 + pe / 100)
    e = validation.error_frame(np.arange(400), forecast, actual, size_class="all")
    t = validation.epi_table(e.iloc[:200], 0.1, 0.9)
    lo, hi = validation.apply_epi(forecast[200:], np.array(["all"] * 200), t)
    assert 0.7 < validation.coverage(actual[200:], lo, hi) < 0.9
