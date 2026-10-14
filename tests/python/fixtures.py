"""Tiny raw files laid out exactly like the real downloads, for parser tests.

Two Wisconsin counties (Dane 55025, Vilas 55125) with made-up numbers. The
layouts follow the agencies' published layouts (column names, header rows,
code values); the values are fake.
"""

import zipfile

import numpy as np
import pandas as pd

COUNTIES = [("55", "025", "Dane County"), ("55", "125", "Vilas County")]


def _age_sex(rng, scale):
    return rng.uniform(0.5, 1.5, size=(2, 18)) * scale


def write_raw(raw, rng=None):
    rng = rng or np.random.default_rng(0)
    pep = raw / "pep"
    pep.mkdir(parents=True)
    base = {c[1]: _age_sex(rng, 1000 if c[1] == "025" else 100) for c in COUNTIES}

    # cc-est2025-alldata-55.csv : YEAR 1..7, AGEGRP 0..18
    rows = []
    for st, co, name in COUNTIES:
        for year in range(1, 8):
            P = base[co] * (1 + 0.01 * year)
            for ag in range(0, 19):
                m = P[0].sum() if ag == 0 else P[0, ag - 1]
                f = P[1].sum() if ag == 0 else P[1, ag - 1]
                rows.append(["050", st, co, "Wisconsin", name, year, ag, m + f, m, f, m * 0.9, f * 0.9])
    cols = ["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "YEAR", "AGEGRP",
            "TOT_POP", "TOT_MALE", "TOT_FEMALE", "WA_MALE", "WA_FEMALE"]
    pd.DataFrame(rows, columns=cols).to_csv(pep / "cc-est2025-alldata-55.csv", index=False)

    # cc-est2020int-alldata-55.csv : YEAR 1..12 (1 = April 2010, 12 = April 2020)
    rows = []
    census2010 = {}
    for st, co, name in COUNTIES:
        for year in range(1, 13):
            P = base[co] * (0.9 + 0.008 * year)
            if year == 1:
                census2010[co] = P
            for ag in range(0, 19):
                m = P[0].sum() if ag == 0 else P[0, ag - 1]
                f = P[1].sum() if ag == 0 else P[1, ag - 1]
                rows.append(["050", st, co, "Wisconsin", name, year, ag, m + f, m, f, m * 0.9, f * 0.9])
    pd.DataFrame(rows, columns=cols).to_csv(pep / "cc-est2020int-alldata-55.csv", index=False)

    # co-est00int-agesex-5yr.csv : SEX 0/1/2, AGEGRP 0..18, one column per date
    rows = []
    date_cols = ["ESTIMATESBASE2000"] + [f"POPESTIMATE{y}" for y in range(2000, 2010)] + \
                ["CENSUS2010POP", "POPESTIMATE2010"]
    for st, co, name in COUNTIES:
        for sex in (0, 1, 2):
            for ag in range(0, 19):
                vals = []
                for c in date_cols:
                    P = census2010[co] if c == "CENSUS2010POP" else base[co] * 0.85
                    v = P.sum(axis=0) if sex == 0 else P[sex - 1]
                    vals.append(v.sum() if ag == 0 else v[ag - 1])
                rows.append(["050", st, co, "Wisconsin", name, sex, ag] + vals)
    pd.DataFrame(rows, columns=["SUMLEV", "STATE", "COUNTY", "STNAME", "CTYNAME", "SEX", "AGEGRP"] + date_cols
                 ).to_csv(pep / "co-est00int-agesex-5yr.csv", index=False)

    # co-est2025-alldata.csv : components; BIRTHS2020 is the partial April-June year
    rows = []
    for st, co, name in COUNTIES:
        rows.append(["050", 2, 3, st, co, "Wisconsin", name, 1000, 1001, 1002, 1003, 1004, 1005, 1006,
                     11, 40, 41, 42, 43, 44])
    pd.DataFrame(rows, columns=["SUMLEV", "REGION", "DIVISION", "STATE", "COUNTY", "STNAME", "CTYNAME",
                                "ESTIMATESBASE2020", "POPESTIMATE2020", "POPESTIMATE2021", "POPESTIMATE2022",
                                "POPESTIMATE2023", "POPESTIMATE2024", "POPESTIMATE2025",
                                "BIRTHS2020", "BIRTHS2021", "BIRTHS2022", "BIRTHS2023", "BIRTHS2024", "BIRTHS2025"]
                 ).to_csv(pep / "co-est2025-alldata.csv", index=False)

    # CO-EST2025-HU.xlsx : title, header rows, ".County, State" rows, footnote
    x = [["table with row headers in column A and column headers in rows 3 through 4", None, None, None, None, None, None, None],
         ["Annual Estimates of Housing Units for Counties in the United States: April 1, 2020 to July 1, 2025 (CO-EST2025-HU)", None, None, None, None, None, None, None],
         ["Geographic Area", "April 1, 2020 Estimates Base", "Housing Unit Estimate (as of July 1)", None, None, None, None, None],
         [None, None, 2020, 2021, 2022, 2023, 2024, 2025],
         ["United States", 140000000, 140100000, 141000000, 142000000, 143000000, 144000000, 145000000],
         [".Dane County, Wisconsin", 240000, 240500, 243000, 246000, 249000, 252000, 255000],
         [".Vilas County, Wisconsin", 26000, 26010, 26100, 26200, 26300, 26400, 26500],
         ["Note: footnote text", None, None, None, None, None, None, None]]
    pd.DataFrame(x).to_excel(pep / "CO-EST2025-HU.xlsx", header=False, index=False)

    # Census API extracts
    api = raw / "census_api"
    api.mkdir()

    def sex_age_cells(P, fmt):
        cells = {}
        from hnf.ages import FEMALE_OFFSET, SEX_BY_AGE_CELLS
        for age, cs in SEX_BY_AGE_CELLS.items():
            for i, c in enumerate(cs):
                cells[fmt(c)] = P[0, age] / len(cs)
                cells[fmt(c + FEMALE_OFFSET)] = P[1, age] / len(cs)
        cells[fmt(1)] = P.sum()
        cells[fmt(2)] = P[0].sum()
        cells[fmt(26)] = P[1].sum()
        return cells

    recs = []
    for st, co, _ in COUNTIES:
        r = {"state": st, "county": co}
        r.update(sex_age_cells(census2010[co], lambda c: f"P012{c:03d}"))
        hh = census2010[co].sum() * 0.4
        r.update({"H003001": hh * 1.1, "H003002": hh, "H003003": hh * 0.1,
                  "H004001": hh, "H004002": hh * 0.4, "H004003": hh * 0.25, "H004004": hh * 0.35})
        r.update({f"H005{c:03d}": hh * 0.01 * c for c in range(1, 9)})
        r.update({f"H017{c:03d}": hh / 10 for c in range(1, 22)})
        recs.append(r)
    pd.DataFrame(recs).to_csv(api / "sf1_2010.csv", index=False)

    recs = []
    for st, co, _ in COUNTIES:
        r = {"state": st, "county": co}
        r.update(sex_age_cells(base[co], lambda c: f"P12_{c:03d}N"))
        hh = base[co].sum() * 0.4
        r.update({"H3_001N": hh * 1.1, "H3_002N": hh, "H3_003N": hh * 0.1})
        r.update({f"H4_{c:03d}N": hh / 4 for c in range(1, 5)})
        r.update({f"H5_{c:03d}N": hh * 0.01 * c for c in range(1, 9)})
        r.update({f"H13_{c:03d}N": hh / 10 for c in range(1, 22)})
        recs.append(r)
    pd.DataFrame(recs).to_csv(api / "dhc_2020.csv", index=False)

    pd.DataFrame([{"state": st, "county": co, "H001001": 900} for st, co, _ in COUNTIES]
                 ).to_csv(api / "sf1_2000.csv", index=False)

    recs = []
    for st, co, _ in COUNTIES:
        r = {"state": st, "county": co}
        P = base[co] * 1.05
        r.update(sex_age_cells(P, lambda c: f"B01001_{c:03d}E"))
        r.update({k.replace("E", "M"): 25 for k in list(r) if k.startswith("B01001_")})
        r["B01001_001M"] = -555555555                      # controlled total: no sampling error
        hh = P.sum() * 0.4
        r.update({f"B25007_{c:03d}E": hh / 10 for c in range(1, 22)})
        r.update({f"B25007_{c:03d}M": 40 for c in range(1, 22)})
        r.update({"B25001_001E": hh * 1.2, "B25001_001M": 50, "B25003_001E": hh, "B25003_001M": 40,
                  "B25003_002E": hh * 0.6, "B25003_002M": 30, "B25003_003E": hh * 0.4, "B25003_003M": 30})
        r.update({f"B25004_{c:03d}E": hh * 0.02 * c for c in range(1, 9)})
        r.update({f"B25004_{c:03d}M": 20 for c in range(1, 9)})
        recs.append(r)
    pd.DataFrame(recs).to_csv(api / "acs5_2024.csv", index=False)

    # Building permits: two header rows, then data rows (30 columns)
    bps = raw / "bps"
    bps.mkdir()
    for y in (2019, 2020):
        lines = ["Survey,FIPS,FIPS,Region,Division,County,,1-unit,,,2-units,,,3-4 units,,,5+ units,,,"
                 "1-unit rep,,,2-units rep,,,3-4 units rep,,,5+units rep,,",
                 "Date,State,County,Code,Code,Name," + ",".join(["Bldgs,Units,Value"] * 8)]
        for st, co, name in COUNTIES:
            vals = [10, 10, 2500000, 1, 2, 300000, 1, 4, 500000, 2, 40, 6000000] * 2
            lines.append(f"{y},{st},{co},2,3,{name} ," + ",".join(str(v) for v in vals))
        (bps / f"co{y}a.txt").write_text("\n".join(lines) + "\n")

    # HUD CHAS Table 8 inside a zip: bands must add to tenure totals
    chas = raw / "chas"
    chas.mkdir()
    t8 = []
    for st, co, name in COUNTIES:
        r = {"source": "2018thru2022", "sumlevel": 50, "geoid": f"05000US{st}{co}",
             "name": f"{name}, Wisconsin", "st": st, "cnty": co}
        r.update({f"T8_est{i}": 0 for i in range(1, 134)})
        owners = {3: 50, 16: 60, 29: 90, 42: 80, 55: 300}
        renters = {69: 120, 82: 70, 95: 60, 108: 30, 121: 40}
        for c, v in {**owners, **renters}.items():
            r[f"T8_est{c}"] = v
        r["T8_est2"] = sum(owners.values())
        r["T8_est68"] = sum(renters.values())
        r["T8_est1"] = r["T8_est2"] + r["T8_est68"]
        t8.append(r)
    with zipfile.ZipFile(chas / "2018thru2022-050-csv.zip", "w") as z:
        z.writestr("050/Table8.csv", pd.DataFrame(t8).to_csv(index=False))

    # NCHS life table spreadsheets (title rows, header, "0–1" ... "100 and over")
    lt_dir = raw / "life_tables" / "us2024"
    lt_dir.mkdir(parents=True)
    x = np.arange(101)
    q = np.clip(0.0005 + 0.00003 * np.exp(0.09 * x), 0, 1)
    q[100] = 1.0
    l = 100000 * np.concatenate([[1], np.cumprod(1 - q[:-1])])
    d = l * q
    L = l - 0.5 * d
    L[100] = l[100] / 0.5
    T = np.cumsum(L[::-1])[::-1]
    for sex in ("M", "F"):
        sheet = [["Table 2. Life table for males: United States, 2024"], [None],
                 ["Age (years)", "Probability of dying between ages x and x + 1", "Number surviving to age x",
                  "Number dying between ages x and x + 1", "Person-years lived between ages x and x + 1",
                  "Total number of person-years lived above age x", "Expectation of life at age x"],
                 [None, "qx", "lx", "dx", "Lx", "Tx", "ex"]]
        for i in range(100):
            sheet.append([f"{i}–{i + 1}", q[i], l[i], d[i], L[i], T[i], T[i] / l[i]])
        sheet.append(["100 and over", 1.0, l[100], d[100], L[100], T[100], T[100] / l[100]])
        sheet.append(["SOURCE: NCHS, National Vital Statistics System, Mortality."])
        pd.DataFrame(sheet).to_excel(lt_dir / f"US_{sex}.xlsx", header=False, index=False)

    # QCEW area slice (annual averages)
    q_dir = raw / "qcew"
    q_dir.mkdir()
    for y in (2019, 2025):
        for st, co, _ in COUNTIES:
            pd.DataFrame([
                {"area_fips": f"{st}{co}", "own_code": 0, "industry_code": "10", "agglvl_code": 70,
                 "size_code": 0, "year": y, "qtr": "A", "disclosure_code": "", "annual_avg_estabs": 100,
                 "annual_avg_emplvl": 5000 + y, "annual_avg_wkly_wage": 1000},
                {"area_fips": f"{st}{co}", "own_code": 5, "industry_code": "10", "agglvl_code": 71,
                 "size_code": 0, "year": y, "qtr": "A", "disclosure_code": "", "annual_avg_estabs": 90,
                 "annual_avg_emplvl": 4000, "annual_avg_wkly_wage": 950},
            ]).to_csv(q_dir / f"{y}_{st}{co}.csv", index=False)
    return {"census2010": census2010, "base": base}
