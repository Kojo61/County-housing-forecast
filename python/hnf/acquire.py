"""Download every public source listed in config/sources.yaml into data/raw/.

Each download is recorded in data/raw/_provenance.csv (URL, UTC time, bytes,
SHA-256). Files already present are skipped unless force=True, so the step
can be re-run safely after an interruption.
"""

import os
import time
from pathlib import Path

import pandas as pd
import requests

from .io import Provenance

STATES = {
    "01": ("AL", "Alabama"), "02": ("AK", "Alaska"), "04": ("AZ", "Arizona"),
    "05": ("AR", "Arkansas"), "06": ("CA", "California"), "08": ("CO", "Colorado"),
    "09": ("CT", "Connecticut"), "10": ("DE", "Delaware"), "11": ("DC", "District of Columbia"),
    "12": ("FL", "Florida"), "13": ("GA", "Georgia"), "15": ("HI", "Hawaii"), "16": ("ID", "Idaho"),
    "17": ("IL", "Illinois"), "18": ("IN", "Indiana"), "19": ("IA", "Iowa"), "20": ("KS", "Kansas"),
    "21": ("KY", "Kentucky"), "22": ("LA", "Louisiana"), "23": ("ME", "Maine"),
    "24": ("MD", "Maryland"), "25": ("MA", "Massachusetts"), "26": ("MI", "Michigan"),
    "27": ("MN", "Minnesota"), "28": ("MS", "Mississippi"), "29": ("MO", "Missouri"),
    "30": ("MT", "Montana"), "31": ("NE", "Nebraska"), "32": ("NV", "Nevada"),
    "33": ("NH", "New Hampshire"), "34": ("NJ", "New Jersey"), "35": ("NM", "New Mexico"),
    "36": ("NY", "New York"), "37": ("NC", "North Carolina"), "38": ("ND", "North Dakota"),
    "39": ("OH", "Ohio"), "40": ("OK", "Oklahoma"), "41": ("OR", "Oregon"),
    "42": ("PA", "Pennsylvania"), "44": ("RI", "Rhode Island"), "45": ("SC", "South Carolina"),
    "46": ("SD", "South Dakota"), "47": ("TN", "Tennessee"), "48": ("TX", "Texas"),
    "49": ("UT", "Utah"), "50": ("VT", "Vermont"), "51": ("VA", "Virginia"),
    "53": ("WA", "Washington"), "54": ("WV", "West Virginia"), "55": ("WI", "Wisconsin"),
    "56": ("WY", "Wyoming"),
}

HEADERS = {"User-Agent": "county-housing-forecast/0.1 (open research code)"}


def state_list(sources):
    s = sources.get("states", "all")
    return sorted(STATES) if s == "all" else [str(x).zfill(2) for x in s]


def download(url, dest, prov, force=False, headers=None, retries=3, timeout=300):
    """Download url to dest. Returns True on success; logs the outcome either way."""
    dest = Path(dest)
    if dest.exists() and not force:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = ""
    for attempt in range(retries):
        try:
            with requests.get(url, headers={**HEADERS, **(headers or {})}, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    prov.log(url, None, status="404")
                    return False
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
            prov.log(url, dest)
            return True
        except requests.RequestException as exc:  # network hiccup: wait and retry
            last = str(exc)
            time.sleep(2 * (attempt + 1))
    prov.log(url, None, status="failed", note=last[:200])
    return False


def acquire_pep(sources, raw, prov, force=False):
    files = sources["pep"]["files"]
    base = sources["pep"]["base"]
    out = Path(raw) / "pep"
    for st in state_list(sources):
        for key in ("cc_est2025_alldata", "cc_est2020int_alldata"):
            url = files[key].format(base=base, st=st)
            download(url, out / Path(url).name, prov, force)
    for key in ("co_est2025_alldata", "co_est00int_agesex", "co_est2020_alldata",
                "co_est2009_alldata", "co_est2025_hu"):
        url = files[key].format(base=base)
        if not download(url, out / Path(url).name, prov, force):
            print(f"  ! could not download {url}\n    get it from the landing page noted in "
                  f"config/sources.yaml and save it to {out}")


def api_variables(pull):
    names = []
    for table, n in pull["tables"].items():
        for cell in range(1, int(n) + 1):
            names.append(pull["naming"].format(table=table, cell=cell))
            if "moe_naming" in pull:
                names.append(pull["moe_naming"].format(table=table, cell=cell))
    return names


def acquire_census_api(sources, raw, prov, force=False, only=None):
    api = sources["census_api"]
    key = os.environ.get(api["key_env"], "")
    if not key:
        print(f"  ! {api['key_env']} is not set. The Census API allows a few keyless calls a day; "
              f"get a free key at https://api.census.gov/data/key_signup.html")
    out = Path(raw) / "census_api"
    out.mkdir(parents=True, exist_ok=True)
    for name, pull in api["pulls"].items():
        if only and name not in only:
            continue
        dest = out / f"{name}.csv"
        if dest.exists() and not force:
            continue
        variables = api_variables(pull)
        merged = None
        for i in range(0, len(variables), 45):            # API limit is 50 variables per call
            chunk = variables[i:i + 45]
            url = f"{api['base_url']}/{pull['dataset']}?get={','.join(chunk)}&{api['geography']}"
            if key:
                url += f"&key={key}"
            r = requests.get(url, headers=HEADERS, timeout=300)
            if r.status_code != 200:
                prov.log(url.replace(key, "KEY") if key else url, None, status=str(r.status_code),
                         note=r.text[:200])
                raise RuntimeError(f"Census API error {r.status_code} for {name}: {r.text[:300]}")
            rows = r.json()
            df = pd.DataFrame(rows[1:], columns=rows[0])
            merged = df if merged is None else merged.merge(df, on=["state", "county"])
            time.sleep(0.5)
        merged.to_csv(dest, index=False)
        prov.log(f"{api['base_url']}/{pull['dataset']} ({len(variables)} variables, all counties)", dest)


def acquire_bps(sources, raw, prov, force=False):
    y0, y1 = sources["bps"]["years"]
    out = Path(raw) / "bps"
    for y in range(y0, y1 + 1):
        url = sources["bps"]["county_annual"].format(year=y)
        if not download(url, out / Path(url).name, prov, force):
            print(f"  ! permits file for {y} not found ({url})")


def acquire_chas(sources, raw, prov, force=False):
    out = Path(raw) / "chas"
    for key, spec in sources["chas"].items():
        if key == "api":
            continue
        url = spec["zip_url"]
        if not download(url, out / Path(url).name, prov, force):
            print(f"  ! CHAS zip not found at {url}.\n    Download '{key}' for County (050) as CSV from "
                  f"https://www.huduser.gov/portal/datasets/cp.html and save the zip in {out}")


def acquire_life_tables(sources, raw, prov, force=False):
    out = Path(raw) / "life_tables"
    for tid, spec in sources["life_tables"].items():
        geos = ["US"] if spec["geo"] == "US" else state_list(sources)
        for geo in geos:
            abbr = "US" if geo == "US" else STATES[geo][0]
            for sex, tab in spec["tables"].items():
                url = spec["url"].format(tab=tab, abbr=abbr)
                dest = out / tid / f"{geo}_{sex}{Path(url).suffix}"
                ok = download(url, dest, prov, force)
                if not ok:   # try common alternatives: .xls, ST-2.xlsx
                    alts = [url[:-1] if url.endswith(".xlsx") else url + "x"]
                    if spec["geo"] != "US":
                        alts.append(url.replace(f"{abbr}{tab}", f"{abbr}-{tab}"))
                    for alt in alts:
                        if download(alt, out / tid / f"{geo}_{sex}{Path(alt).suffix}", prov, force):
                            ok = True
                            break
                if not ok:
                    print(f"  ! life table {tid} {geo} {sex} not found ({url})")


def acquire_qcew(sources, raw, prov, force=False, counties=None):
    q = sources["qcew"]
    out = Path(raw) / "qcew"
    for y in q["years"]:
        if q["mode"] == "singlefile":
            url = q["singlefile"].format(year=y)
            download(url, out / Path(url).name, prov, force, timeout=1200)
        else:
            for fips in counties or []:
                url = q["area_slice"].format(year=y, fips=fips)
                download(url, out / f"{y}_{fips}.csv", prov, force)


def acquire_all(cfg, raw, force=False, steps=None, counties=None):
    src = cfg["sources"]
    prov = Provenance(raw)
    todo = steps or ["pep", "census_api", "bps", "chas", "life_tables", "qcew"]
    if "pep" in todo:
        print("PEP population and housing-unit files ..."); acquire_pep(src, raw, prov, force)
    if "census_api" in todo:
        print("Census API (2000/2010/2020 censuses, ACS) ..."); acquire_census_api(src, raw, prov, force)
    if "bps" in todo:
        print("Building permits ..."); acquire_bps(src, raw, prov, force)
    if "chas" in todo:
        print("HUD CHAS ..."); acquire_chas(src, raw, prov, force)
    if "life_tables" in todo:
        print("NCHS life tables ..."); acquire_life_tables(src, raw, prov, force)
    if "qcew" in todo:
        print("BLS QCEW ..."); acquire_qcew(src, raw, prov, force, counties)
