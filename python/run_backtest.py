"""Step 3: historical back-tests, benchmark comparison and interval calibration.

    python python/run_backtest.py                       # real data in data/processed
    python python/run_backtest.py --data data/synthetic --out output/synthetic

Runs testA10, testA5 and testB5 (config/runs.yaml), or only those named with
--tests (results of tests run earlier are reused from output/<test>/errors.csv),
then:
  * calibrates 5-year empirical prediction intervals on testA5 and checks their
    coverage on the held-out testB5;
  * calibrates 10-year intervals on testA10 (used for the production forecast);
  * writes output/backtests/accuracy_summary.csv, model_vs_benchmarks.csv,
    epi_tables.csv and epi_coverage.csv.
Run this BEFORE run_forecast.py so the forecast can carry calibrated intervals.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hnf import pipeline, validation  # noqa: E402
from hnf.io import load_config  # noqa: E402

TESTS = ["testA10", "testA5", "testB5"]

if __name__ == "__main__":
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(cfg["root"] / "data" / "processed"))
    ap.add_argument("--out", default=str(cfg["root"] / "output"))
    ap.add_argument("--tests", nargs="*", default=TESTS)
    a = ap.parse_args()
    out = Path(a.out) / "backtests"
    out.mkdir(parents=True, exist_ok=True)
    q_lo, q_hi = cfg["model"]["validation"]["epi_quantiles"]

    errors = {}
    for t in a.tests:
        print(f"Running {t} ...")
        try:
            res = pipeline.run(t, cfg, a.data, out_dir=a.out)
        except (FileNotFoundError, KeyError) as exc:
            print(f"  skipped {t}: {exc}")
            continue
        errors[t] = res["errors"]
        e = res["errors"]
        m = validation.summarize(e, ["measure", "method"])
        print(m[m["method"].isin(["model", "AVE", "CONST"])].round(2).to_string(index=False))

    if not errors:
        sys.exit("No back-test could run; check data/processed.")
    # Tests not run this time reuse their saved results, so the tests can be run in
    # stages (A10 and A5 first, freeze the method, then the held-out B5).
    for t in TESTS:
        saved = Path(a.out) / t / "errors.csv"
        if t not in errors and saved.exists():
            errors[t] = pd.read_csv(saved, dtype={"geoid": str})
            print(f"Using saved results for {t} from {saved}")
    allerr = pd.concat(errors.values(), ignore_index=True)
    allerr.to_csv(out / "all_errors.csv", index=False)
    summary = validation.summarize(allerr, ["run", "measure", "method", "horizon"])
    summary.to_csv(out / "accuracy_summary.csv", index=False)
    wins = []
    for t, e in errors.items():
        wins.append(validation.win_rate(e).assign(run=t))
    pd.concat(wins).to_csv(out / "model_vs_benchmarks.csv", index=False)

    # Empirical prediction intervals: calibrate on earlier tests, check on the held-out test.
    tables, cov = [], []
    model_rows = allerr[allerr["method"] == "model"]
    if "testA5" in errors:
        cal = model_rows[(model_rows["run"] == "testA5") & (model_rows["measure"] == "pop")]
        t5 = validation.epi_table(cal, q_lo, q_hi).assign(measure="pop", horizon=5, calibrated_on="testA5")
        tables.append(t5)
        if "testB5" in errors:
            ev = model_rows[(model_rows["run"] == "testB5") & (model_rows["measure"] == "pop")]
            lo, hi = validation.apply_epi(ev["forecast"].to_numpy(), ev["size_class"].to_numpy(), t5)
            cov.append({"measure": "pop", "horizon": 5, "calibrated_on": "testA5", "evaluated_on": "testB5",
                        "nominal": q_hi - q_lo, "coverage": validation.coverage(ev["actual"], lo, hi), "n": len(ev)})
            pooled = model_rows[model_rows["run"].isin(["testA5", "testB5"]) & (model_rows["measure"] == "pop")]
            tables.append(validation.epi_table(pooled, q_lo, q_hi).assign(
                measure="pop", horizon=5, calibrated_on="testA5+testB5 (for production)"))
    if "testA10" in errors:
        for meas in ("pop", "households"):
            cal = model_rows[(model_rows["run"] == "testA10") & (model_rows["measure"] == meas)]
            if len(cal):
                tables.append(validation.epi_table(cal, q_lo, q_hi).assign(
                    measure=meas, horizon=10, calibrated_on="testA10 (for production)"))
    if tables:
        pd.concat(tables).to_csv(out / "epi_tables.csv", index=False)
    if cov:
        pd.DataFrame(cov).to_csv(out / "epi_coverage.csv", index=False)
        print(pd.DataFrame(cov).round(3).to_string(index=False))
    print(f"Back-test outputs written to {out}")
