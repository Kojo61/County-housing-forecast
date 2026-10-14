"""Generate a SYNTHETIC processed dataset for testing the code end to end.

Nothing produced here describes a real place. Counties sit in made-up states
90, 91 and 92 and are named "SYNTHETIC County nnn". The data come from an
annual single-year-of-age simulation (a different structure from the model's
five-year engine), with migration regimes that shift between periods, so
back-tests on this data produce realistic, non-zero errors.

Use it to check that the pipeline runs and that the Python and R engines
agree. Never report its numbers as results.

    python python/make_synthetic.py            # writes data/synthetic/
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnf.ages import HH_GROUPS, HH_LABELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STATES = ["90", "91", "92"]
YEARS = range(1999, 2026)          # July 1 of each year


def life_table(A, B, G, infant_q):
    x = np.arange(101)
    q = 1 - np.exp(-(A + B * np.exp(G * x) * (np.exp(G) - 1) / G))
    q[0] = infant_q
    q[100] = 1.0
    l = np.empty(101)
    l[0] = 100000.0
    for i in range(1, 101):
        l[i] = l[i - 1] * (1 - q[i - 1])
    d = l * q
    L = l - 0.5 * d
    L[0] = l[0] - 0.9 * d[0]
    L[100] = l[100] / (A + B * np.exp(G * 100))      # open interval
    T = np.cumsum(L[::-1])[::-1]
    return pd.DataFrame({"age": x, "qx": q, "lx": l, "dx": d, "Lx": L, "Tx": T, "ex": T / l})


def make_life_tables():
    rows = []
    specs = {"us2010": 2010, "us2024": 2024, "st2019": 2019, "st2022": 2022}
    for tid, year in specs.items():
        geos = ["US"] if tid.startswith("us") else STATES
        improve = 0.985 ** (year - 2010)
        for gi, geo in enumerate(geos):
            state_factor = 1.0 + 0.08 * (gi - 1) if geo != "US" else 1.0
            for sex, (B, infant) in {"M": (4.0e-5, 0.0065), "F": (2.4e-5, 0.0053)}.items():
                lt = life_table(4e-4, B * improve * state_factor, 0.092, infant * improve)
                lt.insert(0, "sex", sex)
                lt.insert(0, "year", year)
                lt.insert(0, "geo", geo)
                lt.insert(0, "table_id", tid)
                rows.append(lt)
    return pd.concat(rows, ignore_index=True)


def migration_profile():
    """Rogers-Castro-like single-year age profile, normalized to mean 1."""
    x = np.arange(101)
    p = 0.3 + 1.6 * np.exp(-0.1 * x) * (x < 15) + 2.2 * np.exp(-((x - 24) / 7.0) ** 2) \
        + 0.4 * np.exp(-((x - 64) / 5.0) ** 2)
    return p / p.mean()


def simulate(rng, n_per_state):
    counties = []
    for st in STATES:
        st_level = rng.normal(0.004, 0.004)
        for j in range(n_per_state):
            geoid = f"{st}{2 * j + 1:03d}"
            size = float(np.exp(rng.uniform(np.log(3000), np.log(1_500_000))))
            # Annual net migration rate regimes for 2000-04, 2005-09, 2010-14, 2015-19, 2020-25
            base = st_level + rng.normal(0, 0.006)
            regimes = base + np.cumsum(rng.normal(0, 0.003, 5))
            counties.append({"geoid": geoid, "state": st, "size": size, "regimes": regimes,
                             "fert": float(np.clip(rng.normal(1.0, 0.12), 0.6, 1.5)),
                             "seasonal": float(rng.choice([0.01, 0.02, 0.03, 0.25], p=[.4, .3, .2, .1])),
                             "other": float(rng.uniform(0.02, 0.07)),
                             "rf": float(np.clip(rng.normal(0.95, 0.15), 0.5, 1.6)),
                             "head": float(rng.normal(1.0, 0.05)),
                             "own": float(rng.normal(0.0, 0.08))})
    lt = life_table(4e-4, 3.2e-5, 0.092, 0.006)
    surv = np.array(lt["Lx"].shift(-1) / lt["Lx"], dtype=float)
    surv[99] = surv[100] = 0.65
    asfr_single = np.zeros(101)
    asfr_single[15:45] = np.interp(np.arange(15, 45), [15, 20, 27, 32, 38, 44], [0.01, 0.06, 0.10, 0.10, 0.05, 0.01])
    prof = migration_profile()
    shape = lt["Lx"].to_numpy() * np.exp(-0.005 * np.arange(101))

    pop_rows, birth_rows = [], []
    for c in counties:
        P = np.stack([shape * 0.51, shape * 0.49]) / shape.sum() * c["size"]   # (2, 101)
        for y in YEARS:
            pop_rows.append((c["geoid"], y, P.copy()))
            if y == YEARS[-1]:
                break
            regime = c["regimes"][min(max((y - 2000) // 5, 0), 4)]
            mig = regime * prof
            nxt = np.zeros_like(P)
            nxt[:, 1:] = P[:, :-1] * surv[None, :-1]
            nxt[:, 100] += P[:, 100] * surv[100]
            births = c["fert"] * (asfr_single * 0.5 * (P[1] + nxt[1])).sum()
            nxt[0, 0] = births * 1.05 / 2.05 * 0.995
            nxt[1, 0] = births / 2.05 * 0.996
            nxt = nxt * (1 + mig[None, :])
            P = np.clip(nxt, 0, None)
            birth_rows.append((c["geoid"], y + 1, births))
    return counties, pop_rows, birth_rows


def to_groups(P):
    g = np.zeros((2, 18))
    for a in range(17):
        g[:, a] = P[:, 5 * a:5 * a + 5].sum(axis=1)
    g[:, 17] = P[:, 85:].sum(axis=1)
    return g


def long_pop(geoid, series, date, G, rng=None, noise=0.0):
    rows = []
    for s, sex in enumerate(("M", "F")):
        for a in range(18):
            v = G[s, a]
            if rng is not None and noise > 0:
                v = v * (1 + rng.normal(0, noise))
            rows.append((geoid, series, date, sex, a, max(v, 0.0)))
    return rows


def main(out_dir, n_per_state=20, seed=20261006):
    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counties, pop_rows, birth_rows = simulate(rng, n_per_state)
    july = {(g, y): to_groups(P) for g, y, P in pop_rows}
    april = {(g, y): 0.25 * july[(g, y - 1)] + 0.75 * july[(g, y)] for g, y, _ in pop_rows if y > 1999}

    rows = []
    for c in counties:
        g = c["geoid"]
        for y in range(2000, 2011):                       # 2000-2010 intercensal
            rows += long_pop(g, "int2000", f"{y}-07-01", july[(g, y)])
        rows += long_pop(g, "int2000", "2000-04-01", april[(g, 2000)])
        rows += long_pop(g, "int2000", "2010-04-01", april[(g, 2010)])
        for y in range(2010, 2020):                       # 2010-2020 intercensal
            rows += long_pop(g, "int2020", f"{y}-07-01", july[(g, y)])
        rows += long_pop(g, "int2020", "2010-04-01", april[(g, 2010)])
        rows += long_pop(g, "int2020", "2020-04-01", april[(g, 2020)])
        rows += long_pop(g, "v2025", "2020-04-01", april[(g, 2020)], rng, 0.002)   # vintage revisions
        for y in range(2020, 2026):
            rows += long_pop(g, "v2025", f"{y}-07-01", july[(g, y)], rng, 0.002)
        rows += long_pop(g, "census2010", "2010-04-01", april[(g, 2010)], rng, 0.001)
        rows += long_pop(g, "census2020", "2020-04-01", april[(g, 2020)], rng, 0.003)
    population = pd.DataFrame(rows, columns=["geoid", "series", "date", "sex", "age", "pop"])
    population.to_csv(out / "population.csv", index=False)

    births = pd.DataFrame(birth_rows, columns=["geoid", "year_end", "births"])
    parts = []
    for series, (a, b) in {"v2009": (2001, 2009), "v2020": (2011, 2020), "v2025": (2021, 2025)}.items():
        d = births[births["year_end"].between(a, b)].copy()
        d["series"] = series
        parts.append(d)
    pd.concat(parts)[["geoid", "series", "year_end", "births"]].to_csv(out / "births.csv", index=False)

    make_life_tables().to_csv(out / "life_tables.csv", index=False)
    shutil.copy(ROOT / "config" / "national_asfr.csv", out / "asfr.csv")

    # Households: headship by age, drifting down slightly for young adults over time.
    base_hr = np.array([0.10, 0.42, 0.52, 0.55, 0.57, 0.58, 0.60, 0.64, 0.62])
    base_own = np.array([0.20, 0.45, 0.62, 0.70, 0.75, 0.78, 0.80, 0.78, 0.70])
    drift = {"sf1_2010": np.array([0.0] * 9), "dhc_2020": np.array([-0.01, -0.02, 0, 0, 0, 0, 0.005, 0.01, 0]),
             "acs5_2024": np.array([-0.015, -0.025, -0.005, 0, 0, 0.005, 0.01, 0.01, 0.005])}
    pop_for = {"sf1_2010": lambda g: april[(g, 2010)], "dhc_2020": lambda g: april[(g, 2020)],
               "acs5_2024": lambda g: np.mean([july[(g, y)] for y in range(2020, 2025)], axis=0)}
    hb_rows, vac_rows, chas_rows = [], [], []
    hh_total = {}
    for c in counties:
        g = c["geoid"]
        for series in ("sf1_2010", "dhc_2020", "acs5_2024"):
            G = pop_for[series](g).sum(axis=0)
            hr = np.clip((base_hr + drift[series]) * c["head"], 0.01, 0.95)
            own = np.clip(base_own + c["own"], 0.05, 0.95)
            tot_hh = 0.0
            for j, label in enumerate(HH_LABELS):
                p = G[HH_GROUPS[label]].sum()
                hh = p * hr[j]
                o, r = hh * own[j], hh * (1 - own[j])
                rec = {"geoid": g, "series": series, "hh_age": label, "pop": p, "owners": o, "renters": r}
                if series == "acs5_2024":                  # survey noise with published SEs
                    cv = 0.6 / np.sqrt(max(p, 50.0))
                    rec.update(pop_se=p * cv * 0.3, owners_se=o * cv * 2, renters_se=r * cv * 2)
                    rec["owners"] = max(o * (1 + rng.normal(0, cv * 2)), 0.0)
                    rec["renters"] = max(r * (1 + rng.normal(0, cv * 2)), 0.0)
                hb_rows.append(rec)
                tot_hh += rec["owners"] + rec["renters"]
            hh_total[(g, series)] = tot_hh
            market_vac = 0.045
            hu = tot_hh / (1 - market_vac - c["seasonal"] - c["other"])
            vac_rows.append({"geoid": g, "source": series, "hu_total": hu, "occupied": tot_hh,
                             "for_rent": hu * market_vac * 0.6, "rented_not_occ": hu * 0.003,
                             "for_sale": hu * market_vac * 0.3, "sold_not_occ": hu * 0.002,
                             "seasonal": hu * c["seasonal"], "migrant": hu * 0.0005,
                             "other_vacant": hu * c["other"]})
        own_share = np.array([0.08, 0.10, 0.15, 0.12, 0.55])
        rent_share = np.array([0.25, 0.20, 0.22, 0.11, 0.22])
        hh = hh_total[(g, "acs5_2024")]
        for tenure, sh in (("owner", own_share), ("renter", rent_share)):
            sh = rng.dirichlet(sh * 60)
            tot = hh * (0.65 if tenure == "owner" else 0.35)
            for b, v in zip(["le30", "30_50", "50_80", "80_100", "gt100"], sh):
                chas_rows.append({"geoid": g, "source": "chas_2018_2022", "tenure": tenure,
                                  "band": b, "households": tot * v})
    pd.DataFrame(hb_rows).to_csv(out / "households_base.csv", index=False)
    pd.DataFrame(vac_rows).to_csv(out / "vacancy.csv", index=False)
    pd.DataFrame(chas_rows).to_csv(out / "chas.csv", index=False)

    # Housing units and permits consistent with household growth and each county's realization factor.
    hu_rows, permit_rows, emp_rows = [], [], []
    for c in counties:
        g = c["geoid"]
        hu2010 = [v for v in vac_rows if v["geoid"] == g and v["source"] == "sf1_2010"][0]["hu_total"]
        hu2020 = [v for v in vac_rows if v["geoid"] == g and v["source"] == "dhc_2020"][0]["hu_total"]
        growth = (hu2020 / hu2010) ** 0.1
        hu2000 = hu2010 / growth ** 10 * (1 + rng.normal(0, 0.01))
        hu_rows += [(g, "census2000", "2000-04-01", hu2000), (g, "census2010", "2010-04-01", hu2010),
                    (g, "census2020", "2020-04-01", hu2020)]
        hu_t = hu2020
        for y in range(2020, 2026):
            hu_rows.append((g, "v2025", f"{y}-07-01", hu_t))
            hu_t *= growth
        for y in range(2000, 2026):
            change = (hu2010 - hu2000) / 10 if y < 2010 else (hu2020 - hu2010) / 10
            units = max(change / c["rf"] * (1 + rng.normal(0, 0.15)), 0.0)
            permit_rows.append({"geoid": g, "year": y, "units": units, "units_1": units * 0.7,
                                "units_2": units * 0.03, "units_34": units * 0.04, "units_5p": units * 0.23})
        # Employment growth leads migration by one period, so the employment add-on has signal.
        emp, wage = c["size"] * 0.45, 800.0
        for y in range(2000, 2026):
            if y in (2005, 2010, 2015, 2019, 2025):
                emp_rows.append({"geoid": g, "year": y, "emp": emp, "wkly_wage": wage})
            p = min(max((y - 2000) // 5, 0), 4)
            emp *= np.exp(0.8 * c["regimes"][min(p + 1, 4)] + rng.normal(0.004, 0.002))
            wage *= np.exp(0.03 + rng.normal(0, 0.005))
    pd.DataFrame(hu_rows, columns=["geoid", "series", "date", "hu"]).to_csv(out / "housing_units.csv", index=False)
    pd.DataFrame(permit_rows).to_csv(out / "permits.csv", index=False)
    pd.DataFrame(emp_rows).to_csv(out / "employment.csv", index=False)
    pd.DataFrame({"geoid": [c["geoid"] for c in counties], "state": [c["state"] for c in counties],
                  "county_name": [f"SYNTHETIC County {c['geoid'][2:]}" for c in counties],
                  "state_name": [f"SYNTHETIC State {c['state']}" for c in counties]}
                 ).to_csv(out / "counties.csv", index=False)
    (out / "README.txt").write_text(
        "SYNTHETIC DATA. Generated by python/make_synthetic.py for software testing only.\n"
        "These files describe no real place and must never be reported as results.\n")
    print(f"Wrote synthetic processed tables for {len(counties)} counties to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--counties-per-state", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261006)
    a = ap.parse_args()
    main(a.out, a.counties_per_state, a.seed)
