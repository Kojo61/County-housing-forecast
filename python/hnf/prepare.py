"""Convert raw downloads (data/raw/) into the processed tables (data/processed/).

Processed tables are the contract between data preparation and the two model
engines (Python and R). Each is a tidy CSV:

population       geoid, series, date, sex, age, pop
births           geoid, series, year_end, births          (July 1 of year_end-1 .. June 30 of year_end)
life_tables      table_id, geo, year, sex, age, qx, lx, dx, Lx, Tx, ex
asfr             year, age, age_label, births_per_1000_women, source
households_base  geoid, series, hh_age, pop, owners, renters [, pop_se, owners_se, renters_se]
vacancy          geoid, source, hu_total, occupied, for_rent, rented_not_occ, for_sale,
                 sold_not_occ, seasonal, migrant, other_vacant
housing_units    geoid, series, date, hu
permits          geoid, year, units, units_1, units_2, units_34, units_5p
chas             geoid, source, tenure, band, households
employment       geoid, year, emp, wkly_wage
counties         geoid, state, county_name, state_name

Every step checks what it reads and writes data/processed/_checks.csv.
"""

import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .ages import FEMALE_OFFSET, HH_GROUPS, SEX_BY_AGE_CELLS
from .io import write_table

CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append({"check": name, "passed": bool(passed), "detail": detail})
    flag = "ok  " if passed else "FAIL"
    print(f"  [{flag}] {name} {detail}")
    return passed


def _geoid(state, county):
    return state.astype(str).str.zfill(2) + county.astype(str).str.zfill(3)


def _read_csv(path, **kw):
    return pd.read_csv(path, encoding="latin-1", low_memory=False, **kw)


# ----------------------------------------------------------------------------
# Population
# ----------------------------------------------------------------------------
def cc_est(paths, year_codes, series):
    """County characteristics files (cc-est2025-alldata, cc-est2020int-alldata)."""
    frames, names = [], []
    codes = {int(k): v for k, v in year_codes.items()}
    for p in paths:
        df = _read_csv(p, usecols=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "YEAR",
                                   "AGEGRP", "TOT_MALE", "TOT_FEMALE"])
        df = df[(df["SUMLEV"].astype(int) == 50) & df["AGEGRP"].between(1, 18)]
        unknown = sorted(set(df["YEAR"].astype(int)) - set(codes))
        if unknown:
            print(f"    note: {Path(p).name} has YEAR codes {unknown} not in config; they are ignored")
        df = df[df["YEAR"].astype(int).isin(codes)]
        df["geoid"] = _geoid(df["STATE"], df["COUNTY"])
        df["date"] = df["YEAR"].astype(int).map(codes)
        names.append(df[["geoid", "CTYNAME", "STNAME"]].drop_duplicates())
        long = df.melt(id_vars=["geoid", "date", "AGEGRP"], value_vars=["TOT_MALE", "TOT_FEMALE"],
                       var_name="sex", value_name="pop")
        long["sex"] = long["sex"].map({"TOT_MALE": "M", "TOT_FEMALE": "F"})
        long["age"] = long["AGEGRP"].astype(int) - 1
        long["series"] = series
        frames.append(long[["geoid", "series", "date", "sex", "age", "pop"]])
    crosswalk = pd.concat(names).drop_duplicates("geoid") if names else None
    return (pd.concat(frames, ignore_index=True) if frames else None), crosswalk


def int2000(path):
    """co-est00int-agesex-5yr.csv: one column per date."""
    df = _read_csv(path)
    df = df[(df["SUMLEV"].astype(int) == 50) & df["SEX"].isin([1, 2]) & df["AGEGRP"].between(1, 18)]
    df["geoid"] = _geoid(df["STATE"], df["COUNTY"])
    date_cols = {}
    for c in df.columns:
        m = re.fullmatch(r"POPESTIMATE(\d{4})", c)
        if m:
            date_cols[c] = f"{m.group(1)}-07-01"
    if "ESTIMATESBASE2000" in df.columns:
        date_cols["ESTIMATESBASE2000"] = "2000-04-01"
    if "CENSUS2010POP" in df.columns:
        date_cols["CENSUS2010POP"] = "2010-04-01"
    long = df.melt(id_vars=["geoid", "SEX", "AGEGRP"], value_vars=list(date_cols),
                   var_name="col", value_name="pop")
    long["date"] = long["col"].map(date_cols)
    long["sex"] = long["SEX"].map({1: "M", 2: "F"})
    long["age"] = long["AGEGRP"].astype(int) - 1
    long["series"] = "int2000"
    long["pop"] = pd.to_numeric(long["pop"], errors="coerce")
    return long[["geoid", "series", "date", "sex", "age", "pop"]]


def births_from_components(path, series, first_full_year):
    df = _read_csv(path)
    df = df[df["SUMLEV"].astype(int) == 50]
    df["geoid"] = _geoid(df["STATE"], df["COUNTY"])
    cols = [c for c in df.columns if re.fullmatch(r"BIRTHS\d{4}", c)]
    long = df.melt(id_vars=["geoid"], value_vars=cols, var_name="col", value_name="births")
    long["year_end"] = long["col"].str[-4:].astype(int)
    long = long[long["year_end"] >= first_full_year]          # drop the partial first year
    long["series"] = series
    return long[["geoid", "series", "year_end", "births"]]


# ----------------------------------------------------------------------------
# Census API extracts
# ----------------------------------------------------------------------------
def _num(df, col, moe=False):
    v = pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)
    if moe:   # ACS special MOE codes: -555555555 = controlled (no sampling error)
        return v.where(v >= 0, np.where(v == -555555555, 0.0, np.nan))
    return v.where(v >= 0)


def sex_age_population(df, name_fn, series, date):
    rows = []
    geoid = _geoid(df["state"], df["county"])
    for age, cells in SEX_BY_AGE_CELLS.items():
        male = sum(_num(df, name_fn(c)) for c in cells)
        female = sum(_num(df, name_fn(c + FEMALE_OFFSET)) for c in cells)
        rows.append(pd.DataFrame({"geoid": geoid, "series": series, "date": date, "sex": "M", "age": age, "pop": male}))
        rows.append(pd.DataFrame({"geoid": geoid, "series": series, "date": date, "sex": "F", "age": age, "pop": female}))
    return pd.concat(rows, ignore_index=True)


def households_by_age(df, pop_name, tenure_name, series, moe_name=None, tenure_moe=None):
    """Population and owner/renter householders by householder age group."""
    geoid = _geoid(df["state"], df["county"])
    rows = []
    for j, (label, ages) in enumerate(HH_GROUPS.items()):
        cells = [c for a in ages for c in SEX_BY_AGE_CELLS[a]]
        cells = cells + [c + FEMALE_OFFSET for c in cells]
        pop = sum(_num(df, pop_name(c)) for c in cells)
        rec = {"geoid": geoid, "series": series, "hh_age": label, "pop": pop,
               "owners": _num(df, tenure_name(3 + j)), "renters": _num(df, tenure_name(13 + j))}
        if moe_name is not None:
            rec["pop_se"] = np.sqrt(sum((_num(df, moe_name(c), moe=True) / 1.645) ** 2 for c in cells))
            rec["owners_se"] = _num(df, tenure_moe(3 + j), moe=True) / 1.645
            rec["renters_se"] = _num(df, tenure_moe(13 + j), moe=True) / 1.645
        rows.append(pd.DataFrame(rec))
    return pd.concat(rows, ignore_index=True)


def vacancy_rows(df, total, occupied, status_name, source):
    geoid = _geoid(df["state"], df["county"])
    names = ["for_rent", "rented_not_occ", "for_sale", "sold_not_occ", "seasonal", "migrant", "other_vacant"]
    out = pd.DataFrame({"geoid": geoid, "source": source,
                        "hu_total": _num(df, total), "occupied": _num(df, occupied)})
    for i, n in enumerate(names):
        out[n] = _num(df, status_name(i + 2))
    return out


def census_api_tables(raw):
    d = Path(raw) / "census_api"
    pop, hb, vac, hu = [], [], [], []
    p = d / "sf1_2010.csv"
    if p.exists():
        df = pd.read_csv(p, dtype=str)
        f = lambda t: (lambda c: f"{t}{c:03d}")
        pop.append(sex_age_population(df, f("P012"), "census2010", "2010-04-01"))
        hb.append(households_by_age(df, f("P012"), f("H017"), "sf1_2010"))
        vac.append(vacancy_rows(df, "H003001", "H003002", f("H005"), "sf1_2010"))
        hu.append(pd.DataFrame({"geoid": _geoid(df["state"], df["county"]), "series": "census2010",
                                "date": "2010-04-01", "hu": _num(df, "H003001")}))
    p = d / "dhc_2020.csv"
    if p.exists():
        df = pd.read_csv(p, dtype=str)
        f = lambda t: (lambda c: f"{t}_{c:03d}N")
        pop.append(sex_age_population(df, f("P12"), "census2020", "2020-04-01"))
        hb.append(households_by_age(df, f("P12"), f("H13"), "dhc_2020"))
        vac.append(vacancy_rows(df, "H3_001N", "H3_002N", f("H5"), "dhc_2020"))
        hu.append(pd.DataFrame({"geoid": _geoid(df["state"], df["county"]), "series": "census2020",
                                "date": "2020-04-01", "hu": _num(df, "H3_001N")}))
    p = d / "sf1_2000.csv"
    if p.exists():
        df = pd.read_csv(p, dtype=str)
        hu.append(pd.DataFrame({"geoid": _geoid(df["state"], df["county"]), "series": "census2000",
                                "date": "2000-04-01", "hu": _num(df, "H001001")}))
    p = d / "acs5_2024.csv"
    if p.exists():
        df = pd.read_csv(p, dtype=str)
        e = lambda t: (lambda c: f"{t}_{c:03d}E")
        m = lambda t: (lambda c: f"{t}_{c:03d}M")
        hb.append(households_by_age(df, e("B01001"), e("B25007"), "acs5_2024",
                                    moe_name=m("B01001"), tenure_moe=m("B25007")))
        vac.append(vacancy_rows(df, "B25001_001E", "B25003_001E", e("B25004"), "acs5_2024"))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else None
    return cat(pop), cat(hb), cat(vac), cat(hu)


# ----------------------------------------------------------------------------
# Housing units (Vintage 2025 Excel table), permits, CHAS, life tables, QCEW
# ----------------------------------------------------------------------------
def hu_v2025(path, crosswalk):
    x = pd.read_excel(path, header=None)
    header_row = None
    for i in range(min(15, len(x))):
        vals = [str(v).strip().split(".")[0] for v in x.iloc[i].tolist()]
        if "2025" in vals:
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not find the year header row in {path}")
    date_for_col = {}
    for j, v in enumerate(x.iloc[header_row].tolist()):
        s = str(v).strip().split(".")[0]
        if re.fullmatch(r"20\d\d", s):
            date_for_col[j] = f"{s}-07-01"
    for i in range(header_row + 1):          # the "Estimates Base" column sits in an upper header row
        for j, v in enumerate(x.iloc[i].tolist()):
            if j > 0 and isinstance(v, str) and "base" in v.lower():
                date_for_col[j] = "2020-04-01"
    names = x[0].astype(str)
    data = x[names.str.startswith(".")].copy()
    data["name"] = data[0].str.lstrip(".").str.strip()
    cw = crosswalk.assign(name=crosswalk["CTYNAME"].str.strip() + ", " + crosswalk["STNAME"].str.strip())
    data = data.merge(cw[["name", "geoid"]], on="name", how="left")
    unmatched = data["geoid"].isna().sum()
    check("housing-unit table names matched to FIPS", unmatched == 0, f"({unmatched} unmatched)")
    rows = []
    for j, date in date_for_col.items():
        rows.append(pd.DataFrame({"geoid": data["geoid"], "series": "v2025", "date": date,
                                  "hu": pd.to_numeric(data[j], errors="coerce")}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["geoid"])


def bps_file(path):
    df = pd.read_csv(path, header=None, skiprows=2, dtype=str, encoding="latin-1")
    df = df[df[1].notna() & df[2].notna()]
    out = pd.DataFrame({
        "geoid": df[1].str.strip().str.zfill(2) + df[2].str.strip().str.zfill(3),
        "year": int(re.search(r"co(\d{4})a", Path(path).name).group(1)),
        "units_1": pd.to_numeric(df[7], errors="coerce"),
        "units_2": pd.to_numeric(df[10], errors="coerce"),
        "units_34": pd.to_numeric(df[13], errors="coerce"),
        "units_5p": pd.to_numeric(df[16], errors="coerce"),
    })
    out["units"] = out[["units_1", "units_2", "units_34", "units_5p"]].sum(axis=1)
    return out


def chas_table(zip_path, spec, source):
    with zipfile.ZipFile(zip_path) as z:
        member = [n for n in z.namelist() if n.lower().endswith(spec["table_file"].lower())]
        if not member:
            raise FileNotFoundError(f"{spec['table_file']} not found inside {zip_path}")
        with z.open(member[0]) as fh:
            df = pd.read_csv(fh, encoding="latin-1", dtype={"st": str, "cnty": str})
    if "st" in df.columns and "cnty" in df.columns:
        geoid = df["st"].str.zfill(2) + df["cnty"].str.zfill(3)
    else:
        geoid = df["geoid"].astype(str).str[-5:]
    cols = spec["columns"]
    rows = []
    sums = {}
    for tenure in ("owner", "renter"):
        total = 0
        for band, col in cols[tenure].items():
            v = pd.to_numeric(df[col], errors="coerce")
            total = total + v
            rows.append(pd.DataFrame({"geoid": geoid, "source": source, "tenure": tenure,
                                      "band": str(band), "households": v}))
        sums[tenure] = (total, pd.to_numeric(df[cols[f"{tenure}_total"]], errors="coerce"))
    for tenure, (s, t) in sums.items():
        bad = (np.abs(s - t) > np.maximum(1, 0.005 * t)).mean()
        if not check(f"CHAS {tenure} bands sum to {tenure} total", bad < 0.01, f"({bad:.1%} of counties off)"):
            raise ValueError("The CHAS column map in config/sources.yaml does not match this file. "
                             "Check the data dictionary in the CHAS zip and fix chas.columns.")
    return pd.concat(rows, ignore_index=True)


AGE_ROW = re.compile(r"^\s*(\d+)\s*(?:[-–—]\s*\d+|\s*(?:and over|and older|\+|or older))", re.I)


def parse_life_table(path):
    """NCHS life-table spreadsheet -> DataFrame age 0..100 with qx, lx, dx, Lx, Tx, ex."""
    x = pd.read_excel(path, header=None)
    rows = []
    for _, r in x.iterrows():
        label = str(r.iloc[0])
        m = AGE_ROW.match(label)
        if not m:
            continue
        nums = pd.to_numeric(pd.Series(r.iloc[1:].tolist()), errors="coerce").dropna().tolist()
        if len(nums) < 6:
            continue
        rows.append([int(m.group(1))] + nums[:6])
    lt = pd.DataFrame(rows, columns=["age", "qx", "lx", "dx", "Lx", "Tx", "ex"]).drop_duplicates("age")
    lt = lt.sort_values("age").reset_index(drop=True)
    if len(lt) != 101 or lt["age"].iloc[0] != 0 or lt["age"].iloc[-1] != 100:
        raise ValueError(f"{path}: expected ages 0..100, found {len(lt)} rows "
                         f"({lt['age'].min() if len(lt) else '-'}..{lt['age'].max() if len(lt) else '-'})")
    return lt


def life_tables(raw, sources):
    frames = []
    for tid, spec in sources["life_tables"].items():
        folder = Path(raw) / "life_tables" / tid
        for path in sorted(folder.glob("*_[MF].xls*")):
            geo, sex = path.stem.split("_")
            lt = parse_life_table(path)
            lt.insert(0, "sex", sex)
            lt.insert(0, "year", spec["year"])
            lt.insert(0, "geo", geo)
            lt.insert(0, "table_id", tid)
            frames.append(lt)
    return pd.concat(frames, ignore_index=True) if frames else None


def qcew(raw, sources):
    folder = Path(raw) / "qcew"
    frames = []
    keep = ["area_fips", "own_code", "industry_code", "agglvl_code", "year",
            "annual_avg_emplvl", "annual_avg_wkly_wage"]
    for z in sorted(folder.glob("*_annual_singlefile.zip")):
        with zipfile.ZipFile(z) as zf:
            with zf.open(zf.namelist()[0]) as fh:
                for chunk in pd.read_csv(fh, usecols=keep, dtype={"area_fips": str, "industry_code": str},
                                         chunksize=500_000):
                    frames.append(chunk[chunk["agglvl_code"] == 70])
    for f in sorted(folder.glob("*_*.csv")):
        df = pd.read_csv(f, usecols=keep, dtype={"area_fips": str, "industry_code": str})
        frames.append(df[(df["own_code"] == 0) & (df["industry_code"] == "10")])
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df = df[df["area_fips"].str.fullmatch(r"\d{5}") & ~df["area_fips"].str.endswith("000")]
    return pd.DataFrame({"geoid": df["area_fips"], "year": df["year"].astype(int),
                         "emp": df["annual_avg_emplvl"], "wkly_wage": df["annual_avg_wkly_wage"]})


# ----------------------------------------------------------------------------
# Geography harmonization and checks
# ----------------------------------------------------------------------------
def harmonize(df, recode, keys, value_cols):
    """Fold historical county codes into current ones (sums the values)."""
    if df is None:
        return None
    df = df.copy()
    df["geoid"] = df["geoid"].replace(recode)
    return df.groupby(keys, as_index=False, dropna=False)[value_cols].sum(min_count=1)


def check_year_codes(population):
    """The intercensal April 2010 / April 2020 codes must match the census counts."""
    tot = population.groupby(["series", "date", "geoid"])["pop"].sum()
    for series, date, census in (("int2020", "2010-04-01", "census2010"),
                                 ("int2000", "2010-04-01", "census2010"),
                                 ("int2020", "2020-04-01", "census2020")):
        try:
            a = tot.loc[(series, date)]
            b = tot.loc[(census, date)]
        except KeyError:
            continue
        both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        diff = (100 * (both["a"] - both["b"]).abs() / both["b"]).median()
        check(f"{series} {date} matches {census}", diff < 1.0,
              f"(median county difference {diff:.3f}%; if this fails, fix pep.year_codes)")


def prepare_all(cfg, raw, processed):
    raw, processed = Path(raw), Path(processed)
    src, model = cfg["sources"], cfg["model"]
    recode = model["geography"]["recode"]
    pep = raw / "pep"
    pops, births = [], []

    print("Population estimates ...")
    v25, cw = cc_est(sorted(pep.glob("cc-est2025-alldata-*.csv")), src["pep"]["year_codes"]["cc_est2025_alldata"], "v2025")
    i20, cw2 = cc_est(sorted(pep.glob("cc-est2020int-alldata-*.csv")), src["pep"]["year_codes"]["cc_est2020int_alldata"], "int2020")
    pops += [p for p in (v25, i20) if p is not None]
    if (pep / "co-est00int-agesex-5yr.csv").exists():
        pops.append(int2000(pep / "co-est00int-agesex-5yr.csv"))
    for fname, series, first in (("co-est2025-alldata.csv", "v2025", 2021),
                                 ("co-est2020-alldata.csv", "v2020", 2011),
                                 ("co-est2009-alldata.csv", "v2009", 2001)):
        if (pep / fname).exists():
            births.append(births_from_components(pep / fname, series, first))

    print("Census API extracts ...")
    pop_c, hb, vac, hu = census_api_tables(raw)
    if pop_c is not None:
        pops.append(pop_c)
    hu_frames = [hu] if hu is not None else []
    if cw is not None and (pep / "CO-EST2025-HU.xlsx").exists():
        hu_frames.append(hu_v2025(pep / "CO-EST2025-HU.xlsx", cw))

    population = harmonize(pd.concat(pops, ignore_index=True) if pops else None, recode,
                           ["geoid", "series", "date", "sex", "age"], ["pop"])
    if population is not None:
        check("population has no negative cells", (population["pop"].dropna() >= 0).all())
        check_year_codes(population)
        write_table(population, processed, "population")
    if births:
        write_table(harmonize(pd.concat(births), recode, ["geoid", "series", "year_end"], ["births"]),
                    processed, "births")
    if hb is not None:
        cols = [c for c in ("pop", "owners", "renters") if c in hb.columns]
        se_cols = [c for c in hb.columns if c.endswith("_se")]
        hbh = harmonize(hb.drop(columns=se_cols), recode, ["geoid", "series", "hh_age"], cols)
        if se_cols:   # standard errors combine in quadrature when counties are merged
            se = hb[["geoid", "series", "hh_age"] + se_cols].copy()
            se["geoid"] = se["geoid"].replace(recode)
            se[se_cols] = se[se_cols] ** 2
            se = se.groupby(["geoid", "series", "hh_age"], as_index=False)[se_cols].sum(min_count=1)
            se[se_cols] = np.sqrt(se[se_cols])
            hbh = hbh.merge(se, on=["geoid", "series", "hh_age"], how="left")
        write_table(hbh, processed, "households_base")
    if vac is not None:
        vcols = [c for c in vac.columns if c not in ("geoid", "source")]
        write_table(harmonize(vac, recode, ["geoid", "source"], vcols), processed, "vacancy")
    if hu_frames:
        write_table(harmonize(pd.concat(hu_frames), recode, ["geoid", "series", "date"], ["hu"]),
                    processed, "housing_units")
    if cw is not None:
        crosswalk = cw.rename(columns={"CTYNAME": "county_name", "STNAME": "state_name"})
        crosswalk["state"] = crosswalk["geoid"].str[:2]
        write_table(crosswalk[["geoid", "state", "county_name", "state_name"]], processed, "counties")

    print("Building permits ...")
    bps = [bps_file(p) for p in sorted((raw / "bps").glob("co*a.txt"))]
    if bps:
        permits = harmonize(pd.concat(bps), recode, ["geoid", "year"],
                            ["units", "units_1", "units_2", "units_34", "units_5p"])
        write_table(permits, processed, "permits")

    print("HUD CHAS ...")
    chas = []
    for key, spec in src["chas"].items():
        if key == "api":
            continue
        for z in sorted((raw / "chas").glob("*.zip")):
            chas.append(chas_table(z, spec, key))
            break
    if chas:
        write_table(harmonize(pd.concat(chas), recode, ["geoid", "source", "tenure", "band"], ["households"]),
                    processed, "chas")

    print("Life tables ...")
    lt = life_tables(raw, src)
    if lt is not None:
        check("life tables start at l0 = 100,000", (lt.loc[lt["age"] == 0, "lx"].round() == 100000).all())
        write_table(lt, processed, "life_tables")

    print("QCEW ...")
    emp = qcew(raw, src)
    if emp is not None:
        emp["geoid"] = emp["geoid"].replace(recode)
        emp = emp.groupby(["geoid", "year"], as_index=False).agg(emp=("emp", "sum"),
                                                                 wkly_wage=("wkly_wage", "mean"))
        write_table(emp, processed, "employment")

    shutil.copy(cfg["root"] / src["asfr_file"], processed / "asfr.csv")
    pd.DataFrame(CHECKS).to_csv(processed / "_checks.csv", index=False)
    failed = [c for c in CHECKS if not c["passed"]]
    print(f"Done: {len(CHECKS) - len(failed)} checks passed, {len(failed)} failed.")
    return failed
