"""Step 1: download every public source into data/raw/ (see config/sources.yaml).

    export CENSUS_API_KEY=...          # free: https://api.census.gov/data/key_signup.html
    python python/run_acquire.py                    # everything
    python python/run_acquire.py --only pep census_api
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnf.acquire import acquire_all  # noqa: E402
from hnf.io import load_config  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(cfg["root"] / "data" / "raw"))
    ap.add_argument("--only", nargs="*", choices=["pep", "census_api", "bps", "chas", "life_tables", "qcew"])
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = ap.parse_args()
    demo = pd.read_csv(cfg["root"] / "config" / "counties_demo.csv", dtype={"geoid": str})["geoid"].tolist()
    acquire_all(cfg, a.raw, force=a.force, steps=a.only, counties=demo)
    print(f"Provenance log: {Path(a.raw) / '_provenance.csv'}")
