"""Step 2: convert data/raw/ into the processed tables in data/processed/.

    python python/run_prepare.py

Writes data/processed/_checks.csv. Stops with a non-zero exit code if any
check fails (for example, a file layout that does not match config).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnf.io import load_config  # noqa: E402
from hnf.prepare import prepare_all  # noqa: E402

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(cfg["root"] / "data" / "raw"))
    ap.add_argument("--processed", default=str(cfg["root"] / "data" / "processed"))
    a = ap.parse_args()
    failed = prepare_all(cfg, a.raw, a.processed)
    sys.exit(1 if failed else 0)
