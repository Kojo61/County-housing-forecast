"""Configuration, processed-table I/O and the download provenance log."""

import csv
import datetime as dt
import hashlib
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

# Columns that must always be read as text (FIPS codes keep their leading zeros).
TEXT_COLUMNS = {"geoid": str, "state": str, "geo": str, "series": str, "date": str,
                "sex": str, "source": str, "table_id": str, "hh_age": str,
                "tenure": str, "band": str, "measure": str}


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(root=ROOT):
    """Return {'model': ..., 'sources': ..., 'runs': ...} from config/."""
    root = Path(root)
    return {
        "model": load_yaml(root / "config" / "model.yaml"),
        "sources": load_yaml(root / "config" / "sources.yaml"),
        "runs": load_yaml(root / "config" / "runs.yaml"),
        "root": root,
    }


def read_table(data_dir, name, required=True):
    """Read data/processed/<name>.csv; return None if optional and missing."""
    path = Path(data_dir) / f"{name}.csv"
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"Processed table '{name}' not found at {path}. Run run_prepare.py first "
                f"(or make_synthetic.py for test data).")
        return None
    head = pd.read_csv(path, nrows=0)
    dtypes = {c: t for c, t in TEXT_COLUMNS.items() if c in head.columns}
    return pd.read_csv(path, dtype=dtypes)


def write_table(df, data_dir, name):
    path = Path(data_dir) / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


class Provenance:
    """Append-only record of every file downloaded: URL, time, size, SHA-256."""

    FIELDS = ["downloaded_utc", "url", "local_path", "bytes", "sha256", "status", "note"]

    def __init__(self, raw_dir):
        self.path = Path(raw_dir) / "_provenance.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow(self.FIELDS)

    def log(self, url, local_path=None, status="ok", note=""):
        size, digest = "", ""
        if local_path is not None and Path(local_path).exists():
            size = Path(local_path).stat().st_size
            digest = sha256(local_path)
        row = [dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), url,
               str(local_path or ""), size, digest, status, note]
        with open(self.path, "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(row)


def years_between(date_a, date_b):
    """Length in years between two ISO dates (365.25-day years)."""
    a = dt.date.fromisoformat(str(date_a))
    b = dt.date.fromisoformat(str(date_b))
    return (b - a).days / 365.25


def add_years(date_iso, years):
    d = dt.date.fromisoformat(str(date_iso))
    return d.replace(year=d.year + int(years)).isoformat()
