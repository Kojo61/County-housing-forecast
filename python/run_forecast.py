"""Step 4: the production forecast (5- and 10-year horizons).

    python python/run_forecast.py                       # real data in data/processed
    python python/run_forecast.py --data data/synthetic --out output/synthetic

Writes output/production/: summary.csv (every county, scenario, horizon and
measure), demo_counties.csv (the demonstration counties, scenarios side by
side), population_age.csv, sensitivity.csv, diagnostics.csv, skipped.csv and
run_manifest.json. If run_backtest.py has been run, population and household
forecasts also carry calibrated 80% empirical prediction intervals.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnf import pipeline, validation  # noqa: E402
from hnf.io import load_config  # noqa: E402


def attach_intervals(summary, diagnostics, epi, cfg):
    """Add epi_low / epi_high columns for base-scenario population and households."""
    v = cfg["model"]["validation"]
    size = dict(zip(diagnostics["geoid"],
                    validation.classify(diagnostics["launch_pop"], v["size_class_breaks"], v["size_class_labels"])))
    out = summary.copy()
    out["epi_low"] = np.nan
    out["epi_high"] = np.nan
    for (measure, horizon), table in epi.groupby(["measure", "horizon"]):
        table = table[table["calibrated_on"].str.contains("production")]
        if table.empty:
            continue
        name = {"pop": "population", "households": "households"}[measure]
        sel = (out["scenario"] == "base") & (out["measure"] == name) & (out["horizon"] == horizon)
        lo, hi = validation.apply_epi(out.loc[sel, "value"].to_numpy(),
                                      out.loc[sel, "geoid"].map(size).to_numpy(), table)
        out.loc[sel, "epi_low"] = lo
        out.loc[sel, "epi_high"] = hi
    return out


if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cfg["root"] / "data" / "processed"))
    ap.add_argument("--out", default=str(cfg["root"] / "output"))
    ap.add_argument("--run", default="production")
    ap.add_argument("--demo", default=str(cfg["root"] / "config" / "counties_demo.csv"))
    a = ap.parse_args()
    demo = pd.read_csv(a.demo, dtype={"geoid": str})["geoid"].tolist()
    res = pipeline.run(a.run, cfg, a.data, out_dir=a.out, demo_counties=demo)
    epi_path = Path(a.out) / "backtests" / "epi_tables.csv"
    if epi_path.exists():
        epi = pd.read_csv(epi_path)
        summary = attach_intervals(res["summary"], res["diagnostics"], epi, cfg)
        summary.to_csv(Path(a.out) / a.run / "summary.csv", index=False)
        print("Attached calibrated 80% prediction intervals from back-tests.")
    else:
        print("No back-test interval tables found; run run_backtest.py first to add intervals.")
    s = res["summary"]
    print(f"Projected {s['geoid'].nunique()} counties; skipped {len(res['skipped'])}. "
          f"Outputs in {Path(a.out) / a.run}")
