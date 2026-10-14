"""Run a forecast or a back-test from a run specification (config/runs.yaml).

The production forecast and every back-test go through run(): the only
difference between them is the specification. Computation is national (all
counties with complete inputs), because small-county shrinkage borrows
strength from every county in the state; the demonstration counties are a
filtered view of the national output.
"""

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import (benchmarks, employment, fertility, gap, households, income, migration,
               mortality, projection, supply, validation, __version__)
from .ages import AGE_LABELS, FERT_IDX, SEXES
from .io import add_years, read_table, sha256, years_between, write_table

LEVELS = ("low", "base", "high")


# ----------------------------------------------------------------------------
# Input helpers
# ----------------------------------------------------------------------------
class Inputs:
    """Processed tables, each loaded once."""

    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self._cache = {}

    def get(self, name, required=True):
        if name not in self._cache:
            self._cache[name] = read_table(self.dir, name, required)
        return self._cache[name]


def pop_array(pop, series, date, geoids):
    """(C, 2, 18) array for one series and date; NaN where missing."""
    d = pop[(pop["series"] == series) & (pop["date"] == date)]
    if d.empty:
        raise KeyError(f"No population for series={series} date={date} in processed/population.csv")
    arr = np.full((len(geoids), 2, 18), np.nan)
    ix = pd.Index(geoids).get_indexer(d["geoid"])
    ok = ix >= 0
    s = (d["sex"].to_numpy() == "F").astype(int)
    a = d["age"].to_numpy(int)
    arr[ix[ok], s[ok], a[ok]] = d["pop"].to_numpy(float)[ok]
    return arr


def nearest_date(pop, series, target):
    dates = sorted(pop.loc[pop["series"] == series, "date"].unique())
    if not dates:
        raise KeyError(f"No dates for population series {series}")
    t = dt.date.fromisoformat(target)
    return min(dates, key=lambda d: abs((dt.date.fromisoformat(d) - t).days))


def series_value(df, geoids, series, value, date=None):
    d = df[df["series"] == series]
    if date is None:
        dates = d["date"].unique() if "date" in d.columns else [None]
        if len(dates) > 1:
            raise ValueError(f"Series {series} has several dates {list(dates)}; give one in runs.yaml")
    else:
        d = d[d["date"] == date]
    s = d.groupby("geoid")[value].sum(min_count=1)
    return np.array(s.reindex(geoids), dtype=float)


def births_window(births, geoids, series, y0, y1):
    d = births[(births["series"] == series) & births["year_end"].between(y0, y1)]
    s = d.groupby("geoid")["births"].sum()
    n_years = d.groupby("geoid")["year_end"].nunique().reindex(geoids)
    out = np.array(s.reindex(geoids), dtype=float)
    out[np.array(n_years, dtype=float) != (y1 - y0 + 1)] = np.nan   # incomplete windows are unknown
    return out


def asfr_vector(asfr, year):
    d = asfr[asfr["year"] == year].sort_values("age")
    if d.empty:
        raise KeyError(f"No national ASFR for {year} in config/national_asfr.csv")
    v = np.zeros(len(FERT_IDX))
    for a, r in zip(d["age"], d["births_per_1000_women"]):
        v[int(a) - FERT_IDX[0]] = r / 1000.0
    return v


def permits_by_year(permits, geoids, y0, y1):
    """(C, Y) permitted units; counties absent from a year's file have 0."""
    d = permits[permits["year"].between(y0, y1)]
    wide = d.pivot_table(index="geoid", columns="year", values="units", aggfunc="sum")
    wide = wide.reindex(index=geoids, columns=range(y0, y1 + 1))
    years_present = [y for y in range(y0, y1 + 1) if y in set(d["year"])]
    if len(years_present) < (y1 - y0 + 1):
        missing = sorted(set(range(y0, y1 + 1)) - set(years_present))
        raise KeyError(f"Permit years missing from processed/permits.csv: {missing}")
    return wide.fillna(0.0).to_numpy(float)


def vacancy_arrays(vac, geoids, source):
    d = vac[vac["source"] == source].set_index("geoid").reindex(geoids)
    cols = ["hu_total", "occupied", "seasonal", "migrant", "other_vacant"]
    return {c: d[c].to_numpy(float) for c in cols}


# ----------------------------------------------------------------------------
# The run
# ----------------------------------------------------------------------------
def run(name, cfg, data_dir, out_dir=None, demo_counties=None):
    spec = cfg["runs"][name]
    model = cfg["model"]
    inp = Inputs(data_dir)
    pop = inp.get("population")
    life = inp.get("life_tables")
    asfr_tab = inp.get("asfr")
    lookup = mortality.survival_lookup(life)
    srb = model["population"]["sex_ratio_at_birth"]
    launch = spec["launch"]
    steps = int(spec["steps"])

    # ---- county universe and launch population --------------------------------
    excluded = set(model["geography"]["exclude_states"])
    in_launch = pop[(pop["series"] == launch["series"]) & (pop["date"] == launch["date"])]
    geoids = sorted(g for g in in_launch["geoid"].unique() if g[:2] not in excluded)
    masks = {}
    P0 = pop_array(pop, launch["series"], launch["date"], geoids)
    masks["launch population"] = np.isfinite(P0).all(axis=(1, 2)) & (np.nansum(P0, axis=(1, 2)) > 0)

    windows_raw = []
    for w in spec["migration_windows"]:
        Ps = pop_array(pop, w["start"]["series"], w["start"]["date"], geoids)
        Pe = pop_array(pop, w["end"]["series"], w["end"]["date"], geoids)
        masks[f"window {w['id']}"] = np.isfinite(Ps).all(axis=(1, 2)) & np.isfinite(Pe).all(axis=(1, 2))
        b = None
        if "births" in w:
            bt = births_window(inp.get("births"), geoids, w["births"]["series"], *w["births"]["years"])
            masks[f"births {w['id']}"] = np.isfinite(bt)
            share_m = srb / (1 + srb)
            b = np.stack([bt * share_m, bt * (1 - share_m)], axis=1)
        windows_raw.append((w, Ps, Pe, b))

    keep = np.logical_and.reduce(list(masks.values()))
    skipped = pd.DataFrame(
        [{"geoid": g, "reason": "; ".join(k for k, m in masks.items() if not m[i])}
         for i, g in enumerate(geoids) if not keep[i]], columns=["geoid", "reason"])
    geoids = [g for g, k in zip(geoids, keep) if k]
    P0 = P0[keep]
    states = np.array([g[:2] for g in geoids])
    C = len(geoids)
    if C == 0:
        raise RuntimeError("No county has complete inputs for this run; see skipped list.")

    # ---- migration ----------------------------------------------------------------
    K_mig = model["migration"]["shrinkage_K"]
    beta = model["population"]["births_migration_factor"]
    win_res = {}
    for (w, Ps, Pe, b) in windows_raw:
        S_w, Sb_w = mortality.county_survival(states, w["life_table"], lookup)
        yrs = years_between(w["start"]["date"], w["end"]["date"])
        win_res[w["id"]] = migration.window_rates(
            Ps[keep], Pe[keep], S_w, Sb_w, yrs, states, K_mig, beta,
            births=None if b is None else b[keep])
    ids = [w["id"] for w in spec["migration_windows"]]
    weights = [spec.get("migration_weights", {}).get(i, 1.0) for i in ids]
    sched = migration.scenario_schedules([win_res[i] for i in ids], weights, P0)
    m_state_base = np.tensordot(np.asarray(weights) / np.sum(weights),
                                np.stack([win_res[i]["m_state"] for i in ids]), axes=1)

    # ---- optional employment-informed level --------------------------------------
    emp_table = None
    use_emp = ("employment" in spec) and (name != "production" or model["employment"]["use_in_production"])
    if "employment" in spec:
        emp_df = inp.get("employment", required=False)
        if emp_df is not None:
            e = spec["employment"]
            tr, ap = e["train"], e["apply"]
            feats = lambda yrs, prior: {
                "emp_growth": employment.annual_log_growth(emp_df, geoids, yrs[0], yrs[1], "emp"),
                "wage_growth": employment.annual_log_growth(emp_df, geoids, yrs[0], yrs[1], "wkly_wage"),
                "prior_mig": win_res[prior]["crude"]}
            pred, emp_table = employment.fit_and_predict(
                win_res[tr["target_window"]]["crude"], feats(tr["emp_years"], tr["prior_window"]),
                feats(ap["emp_years"], ap["prior_window"]), states)
            base_crude = migration.implied_net_migrants(P0, sched["base"][0]) / P0.sum(axis=(1, 2))
            delta = np.where(np.isfinite(pred), pred - base_crude, 0.0)
            if use_emp:
                sched["econ"] = migration.shift_schedule(*sched["base"], P0, m_state_base, delta)
        else:
            use_emp = False

    # ---- fertility -----------------------------------------------------------------
    f = spec["fertility"]
    asfr = asfr_vector(asfr_tab, f["asfr_year"])
    fi = f["index"]
    y0, y1 = fi["years"]
    births_obs = births_window(inp.get("births"), geoids, fi["births_series"], y0, y1)
    women = np.stack([pop_array(pop, fi["pop_series"], nearest_date(pop, fi["pop_series"], f"{y - 1}-07-01"),
                                geoids)[:, 1, FERT_IDX] for y in range(y0, y1 + 1)])
    k, k_raw, k_w = fertility.fertility_index(
        births_obs, fertility.expected_births(women, asfr), states, model["fertility"]["shrinkage_K"])

    # ---- population projection ------------------------------------------------------
    S, Sb = mortality.county_survival(states, spec["projection_life_table"], lookup)
    floor = model["migration"]["min_survival_plus_migration"]
    mig_levels = list(LEVELS) + (["econ"] if "econ" in sched else [])
    pops = {}
    for lvl in mig_levels:
        m, mb = sched[lvl]
        pops[lvl], _ = projection.project(P0, S, Sb, m, mb, asfr, k, steps, srb=srb, floor=floor)
    dates = [add_years(launch["date"], 5 * s) for s in range(steps + 1)]

    # ---- households, income bands, supply and gap ----------------------------------
    hh_res, hr_sc = None, None
    if "households" in spec:
        hb = inp.get("households_base")
        hcfg = spec["households"]
        base_rates = households.rates(hb[hb["series"] == hcfg["base"]], geoids, states,
                                      shrink=model["households"]["shrink_acs_rates"])
        alts = [households.rates(hb[hb["series"] == a], geoids, states,
                                 shrink=model["households"]["shrink_acs_rates"])["hr"]
                for a in hcfg.get("alternatives", [])]
        hr_sc = households.scenario_rates(base_rates["hr"], alts) if alts else \
            {lvl: base_rates["hr"] for lvl in LEVELS}
        # Scenario headship rates phase in linearly: every scenario starts from the base
        # rates at launch (current households are not in doubt) and reaches its own
        # rates at the final horizon.
        hh_res = {}
        for ml in mig_levels:
            for hl in LEVELS:
                hh_res[(ml, hl)] = [
                    households.project(p, base_rates["hr"] + (s / steps) * (hr_sc[hl] - base_rates["hr"]),
                                       base_rates["own"])
                    for s, p in enumerate(pops[ml])]

    shares = None
    if spec.get("chas"):
        ch = inp.get("chas")
        shares = income.shares(ch[ch["source"] == spec["chas"]], geoids, model["income_bands"])

    sup = None
    if "supply" in spec:
        sp = spec["supply"]
        hu = inp.get("housing_units")
        permits = inp.get("permits")
        rz = sp["realization"]
        hu_start = series_value(hu, geoids, rz["start"]["series"], "hu", rz["start"].get("date"))
        hu_end = series_value(hu, geoids, rz["end"]["series"], "hu", rz["end"].get("date"))
        p_rz = permits_by_year(permits, geoids, *rz["permit_years"]).sum(axis=1)
        rf = supply.realization_factor(hu_start, hu_end, p_rz, states,
                                       model["supply"]["shrinkage_K"], model["supply"]["bounds"])
        rz_years = rz["permit_years"][1] - rz["permit_years"][0] + 1
        trend_per_year = (hu_end - hu_start) / rz_years
        p_proj = permits_by_year(permits, geoids, *sp["projection_permit_years"])
        annual = {"low": p_proj.min(axis=1), "base": p_proj.mean(axis=1), "high": p_proj.max(axis=1)}
        hu0 = series_value(hu, geoids, sp["base_hu"]["series"], "hu", sp["base_hu"].get("date"))

        def additions(level, years):
            permitted = rf["rf"] * annual[level] * years
            # Counties with no permit coverage in the estimation window follow their census trend.
            return np.where(p_rz > 0, permitted, trend_per_year * years)

        sup = {"rf": rf, "hu0": hu0, "additions": additions, "annual": annual, "p_rz": p_rz,
               "trend_per_year": trend_per_year}
        if "conditional_permit_years" in sp:
            p_cond = permits_by_year(permits, geoids, *sp["conditional_permit_years"])
            sup["conditional_annual"] = p_cond.mean(axis=1)

    avail = None
    if spec.get("vacancy"):
        vac = vacancy_arrays(inp.get("vacancy"), geoids, spec["vacancy"])
        avail = supply.available_share(vac, model["vacancy"]["other_vacant_available_share"])

    # ---- assemble outputs ----------------------------------------------------------
    rows = []

    def add(geo_vals, date, horizon, scenario, measure, values):
        rows.append(pd.DataFrame({"run": name, "geoid": geo_vals, "date": date, "horizon": horizon,
                                  "scenario": scenario, "measure": measure, "value": values}))

    vac_cfg = model["vacancy"]
    composite = model["scenarios"]["composite"]

    def scenario_block(choice):
        """Totals for one combination of assumptions at every step -> {measure: [(C,) per step]}."""
        ml, hl, vl, sl = choice["migration"], choice["headship"], choice["vacancy"], choice["supply"]
        out = {"population": [p.sum(axis=(1, 2)) for p in pops[ml]]}
        if hh_res is not None:
            hs = hh_res[(ml, hl)]
            out["households"] = [h["households"].sum(axis=1) for h in hs]
            out["owners"] = [h["owners"].sum(axis=1) for h in hs]
            out["renters"] = [h["renters"].sum(axis=1) for h in hs]
            v_o, v_r = vac_cfg["target_owner"][vl], vac_cfg["target_renter"][vl]
            req = [gap.required_units(o, r, v_o, v_r) for o, r in zip(out["owners"], out["renters"])]
            if shares is not None:
                for t_i, t in enumerate(income.TENURES):
                    for b_i, b in enumerate(model["income_bands"]):
                        out[f"{t}_{b}"] = [income.by_band(o, r, shares)[:, t_i, b_i]
                                           for o, r in zip(out["owners"], out["renters"])]
                for b in model["income_bands"]:
                    out[f"hh_{b}"] = [out[f"owner_{b}"][s] + out[f"renter_{b}"][s] for s in range(steps + 1)]
                out["hh_gt80"] = [out["hh_80_100"][s] + out["hh_gt100"][s] for s in range(steps + 1)]
                for b in model["income_bands"]:
                    out[f"required_units_{b}"] = [
                        out[f"owner_{b}"][s] / (1 - v_o) + out[f"renter_{b}"][s] / (1 - v_r)
                        for s in range(steps + 1)]
            if sup is not None and avail is not None:
                for key in ("required_units", "stock", "available_units", "gap",
                            "existing_gap", "growth_in_need", "pipeline"):
                    out[key] = []
                for s in range(steps + 1):
                    comp = gap.gap_components(req[s], req[0], sup["hu0"],
                                              sup["additions"](sl, 5 * s), avail)
                    for key, v in comp.items():
                        out[key].append(v)
        elif sup is not None:
            out["stock"] = [sup["hu0"] + sup["additions"](sl, 5 * s) for s in range(steps + 1)]
        return out

    blocks = {sc: scenario_block(choice) for sc, choice in composite.items()}
    if "econ" in mig_levels:
        blocks["econ"] = scenario_block({"migration": "econ", "headship": "base",
                                         "vacancy": "base", "supply": "base"})
    for sc, block in blocks.items():
        for measure, series in block.items():
            for s, vals in enumerate(series):
                add(geoids, dates[s], 5 * s, sc, measure, vals)

    # One-at-a-time sensitivity of the final-step gap (others at base).
    sens_rows = []
    if hh_res is not None and sup is not None and avail is not None:
        base_choice = composite["base"]
        base_gap = blocks["base"]["gap"][-1]
        for dim in ("migration", "headship", "vacancy", "supply"):
            for lvl in ("low", "high"):
                ch = dict(base_choice, **{dim: lvl})
                g = scenario_block(ch)["gap"][-1]
                sens_rows.append(pd.DataFrame({"run": name, "geoid": geoids, "date": dates[-1],
                                               "assumption": dim, "level": lvl,
                                               "gap": g, "change_from_base": g - base_gap}))

    results = {"summary": pd.concat(rows, ignore_index=True), "skipped": skipped}
    if sens_rows:
        results["sensitivity"] = pd.concat(sens_rows, ignore_index=True)

    # Population by age and sex for each migration scenario.
    age_rows = []
    for lvl in mig_levels:
        for s, p in enumerate(pops[lvl]):
            for si, sex in enumerate(SEXES):
                for a in range(18):
                    age_rows.append(pd.DataFrame({"run": name, "geoid": geoids, "date": dates[s],
                                                  "migration": lvl, "sex": sex, "age": a,
                                                  "age_label": AGE_LABELS[a], "pop": p[:, si, a]}))
    results["population_age"] = pd.concat(age_rows, ignore_index=True)

    # Diagnostics: every estimated parameter, raw and shrunk, so reviewers can audit it.
    diag = pd.DataFrame({"geoid": geoids, "state": states, "launch_pop": P0.sum(axis=(1, 2)),
                         "fertility_index_raw": k_raw, "fertility_index": k, "fertility_weight": k_w})
    for i in ids:
        r = win_res[i]
        diag[f"mig_crude_{i}"] = r["crude"]
        diag[f"mig_shrink_weight_mean_{i}"] = np.nanmean(r["w"], axis=(1, 2))
    for lvl in ("low", "base", "high"):
        diag[f"mig_crude_{lvl}"] = migration.implied_net_migrants(P0, sched[lvl][0]) / P0.sum(axis=(1, 2))
    if "econ" in sched:
        diag["mig_crude_econ"] = migration.implied_net_migrants(P0, sched["econ"][0]) / P0.sum(axis=(1, 2))
    if hh_res is not None:
        diag["headship_total_launch"] = blocks["base"]["households"][0] / P0.sum(axis=(1, 2))
    if sup is not None:
        diag["rf_raw"] = sup["rf"]["rf_raw"]
        diag["rf"] = sup["rf"]["rf"]
        diag["rf_state"] = sup["rf"]["rf_state"]
        diag["rf_weight"] = sup["rf"]["w"]
        lo_r, hi_r = model["supply"]["review_range"]
        diag["rf_review_flag"] = (sup["rf"]["rf"] < lo_r) | (sup["rf"]["rf"] > hi_r)
        diag["permits_rf_window"] = sup["p_rz"]
        diag["permits_annual_base"] = sup["annual"]["base"]
    if avail is not None:
        diag["available_share"] = avail
    results["diagnostics"] = diag
    if emp_table is not None:
        results["employment_model"] = emp_table

    # ---- back-test scoring -----------------------------------------------------------
    if "targets" in spec:
        results["errors"] = score(spec, name, inp, geoids, states, P0, pops, blocks, sup, model, dates)

    # ---- write ----------------------------------------------------------------------
    if out_dir is not None:
        write_outputs(results, name, spec, cfg, data_dir, out_dir, demo_counties)
    return results


# ----------------------------------------------------------------------------
# Back-test scoring
# ----------------------------------------------------------------------------
def score(spec, name, inp, geoids, states, P0, pops, blocks, sup, model, dates):
    pop = inp.get("population")
    vcfg = model["validation"]
    hist = spec["history"]
    p_long = pop_array(pop, hist[0]["series"], hist[0]["date"], geoids).sum(axis=(1, 2))
    p_short = pop_array(pop, hist[1]["series"], hist[1]["date"], geoids).sum(axis=(1, 2))
    p_launch = P0.sum(axis=(1, 2))
    yrs_long = years_between(hist[0]["date"], spec["launch"]["date"])
    yrs_short = years_between(hist[1]["date"], spec["launch"]["date"])
    growth = 100 * (np.log(p_launch / p_long) / yrs_long)
    size_cls = validation.classify(p_launch, vcfg["size_class_breaks"], vcfg["size_class_labels"])
    growth_cls = validation.classify(growth, vcfg["growth_class_breaks"], vcfg["growth_class_labels"])

    frames = []

    def add(measure, method, forecast, actual, horizon):
        frames.append(validation.error_frame(geoids, forecast, actual, run=name, measure=measure,
                                             method=method, horizon=horizon, size_class=size_cls,
                                             growth_class=growth_cls, launch_pop=p_launch))

    for t in spec["targets"]:
        step = int(t["step"])
        h = 5 * step
        measure = t["measure"]
        method_suffix = "_diagnostic" if t.get("diagnostic") else ""
        if t["source"] == "population":
            actual = pop_array(pop, t["series"], t["date"], geoids).sum(axis=(1, 2))
            add(measure, "model" + method_suffix, blocks["base"]["population"][step], actual, h)
            if "econ" in blocks:
                add(measure, "model_econ", blocks["econ"]["population"][step], actual, h)
            bm = benchmarks.population_benchmarks(p_long, p_short, p_launch, yrs_long, yrs_short, h, states)
            for k, v in bm.items():
                add(measure, k, v, actual, h)
        elif t["source"] == "households_base":
            hb = inp.get("households_base")
            d = hb[hb["series"] == t["series"]].copy()
            d["households"] = d["owners"] + d["renters"]
            actual = d.groupby("geoid")[measure].sum().reindex(geoids).to_numpy(float)
            add(measure, "model", blocks["base"][measure][step], actual, h)
            # Benchmark: constant household-to-population ratio applied to the model's population.
            ratio0 = blocks["base"][measure][0] / p_launch
            add(measure, "CONST_RATIO", ratio0 * blocks["base"]["population"][step], actual, h)
        elif t["source"] == "housing_units":
            hu = inp.get("housing_units")
            actual = series_value(hu, geoids, t["series"], "hu", t.get("date"))
            add(measure, "model" + method_suffix, blocks["base"]["stock"][step], actual, h)
            if sup is not None:
                add(measure, "CONST", sup["hu0"], actual, h)
                lin = sup["hu0"] + sup["trend_per_year"] * h
                add(measure, "LIN", np.maximum(lin, 0), actual, h)
                if "conditional_annual" in sup:
                    cond = sup["hu0"] + np.where(sup["p_rz"] > 0,
                                                 sup["rf"]["rf"] * sup["conditional_annual"] * h,
                                                 sup["trend_per_year"] * h)
                    add(measure, "model_conditional_permits", cond, actual, h)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------------
def write_outputs(results, name, spec, cfg, data_dir, out_dir, demo_counties):
    out = Path(out_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    for key, df in results.items():
        df.to_csv(out / f"{key}.csv", index=False)
    if "errors" in results:
        e = results["errors"]
        metrics = pd.concat([
            validation.summarize(e, ["measure", "method"]).assign(group_type="all", group="all"),
            validation.summarize(e, ["measure", "method", "size_class"]).rename(
                columns={"size_class": "group"}).assign(group_type="size_class"),
            validation.summarize(e, ["measure", "method", "growth_class"]).rename(
                columns={"growth_class": "group"}).assign(group_type="growth_class"),
        ], ignore_index=True)
        metrics.to_csv(out / "metrics.csv", index=False)
        validation.win_rate(e).to_csv(out / "model_vs_benchmarks.csv", index=False)
    if demo_counties is not None:
        demo = results["summary"][results["summary"]["geoid"].isin(demo_counties)]
        wide = demo.pivot_table(index=["geoid", "date", "horizon", "measure"],
                                columns="scenario", values="value").reset_index()
        wide.to_csv(out / "demo_counties.csv", index=False)
    processed = Path(data_dir)
    manifest = {
        "run": name,
        "model_version": __version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "spec": spec,
        "model_config": cfg["model"],
        "inputs_sha256": {p.name: sha256(p) for p in sorted(processed.glob("*.csv"))},
        "counties_projected": int(results["summary"]["geoid"].nunique()),
        "counties_skipped": int(len(results["skipped"])),
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
