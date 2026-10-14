"""Download step against a mocked network: every URL is built and every file lands."""

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
from hnf import acquire  # noqa: E402
from hnf.io import load_config  # noqa: E402


class FakeResponse:
    def __init__(self, url):
        self.url = url
        self.status_code = 200
        self.text = ""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        yield f"fake content of {self.url}".encode()

    def json(self):
        query = self.url.split("get=")[1].split("&")[0]
        names = query.split(",")
        return [names + ["state", "county"], ["1"] * len(names) + ["55", "025"]]


def test_acquire_builds_every_url_and_logs_provenance(tmp_path, monkeypatch):
    calls = []

    def fake_get(url, **kw):
        calls.append(url)
        return FakeResponse(url)

    monkeypatch.setattr(acquire.requests, "get", fake_get)
    monkeypatch.setattr(acquire.time, "sleep", lambda s: None)
    cfg = load_config()
    cfg = copy.deepcopy(cfg)
    cfg["sources"]["states"] = ["55"]
    acquire.acquire_all(cfg, tmp_path, counties=["55025"])

    files = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert "pep/cc-est2025-alldata-55.csv" in files
    assert "pep/cc-est2020int-alldata-55.csv" in files
    assert "pep/CO-EST2025-HU.xlsx" in files
    assert "bps/co2000a.txt" in files and "bps/co2025a.txt" in files
    assert "life_tables/st2022/55_M.xlsx" in files and "life_tables/us2010/US_F.xlsx" in files
    assert "chas/2018thru2022-050-csv.zip" in files
    assert "qcew/2025_annual_singlefile.zip" in files
    acs = pd.read_csv(tmp_path / "census_api" / "acs5_2024.csv", dtype=str)
    assert "B25007_021M" in acs.columns and "B01001_049E" in acs.columns
    assert any("NVSR/74-12/WI2.xlsx" in u for u in calls)
    assert all("for=county:*&in=state:*" in u for u in calls if "api.census.gov" in u)
    prov = pd.read_csv(tmp_path / "_provenance.csv")
    assert len(prov) >= 40 and (prov["status"] == "ok").all() and prov["sha256"].notna().all()
