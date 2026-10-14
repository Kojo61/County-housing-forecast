# U.S. County Housing Needs Forecasting Model — Version 0.1

An open, reproducible model that forecasts, for every U.S. county, 5- and 10-year
population, households by tenure and HUD income band, the housing units those
households require, the expected supply from the authorized construction pipeline,
and the resulting gap — with low/base/high scenarios, empirical prediction
intervals, and historical back-tests against simple benchmarks.

The same equations run in **Python** (`python/hnf/`) and **R** (`R/hnf_engine.R`);
`R/crosscheck.R` confirms both engines produce identical numbers.

> **Status.** Code, configuration and tests are complete for Version 0.1. No real
> results are included: every number in a report must come from running the
> pipeline on the downloaded public data (steps 1–4 below). The files in
> `data/synthetic/` are made-up test data and must never be reported.

---

## Quick start: prove the code runs (2 minutes, no downloads)

```bash
pip install -r requirements.txt
python python/make_synthetic.py                                   # synthetic test data
python python/run_backtest.py --data data/synthetic --out output/synthetic
python python/run_forecast.py --data data/synthetic --out output/synthetic
Rscript R/crosscheck.R data/synthetic output/synthetic            # R and Python must agree
python -m pytest tests/python -q && Rscript tests/R/test_engine.R
```

R needs the `data.table` and `yaml` packages (`testthat` for the tests).

## Real run

```bash
export CENSUS_API_KEY=your_key        # free: https://api.census.gov/data/key_signup.html
python python/run_acquire.py          # 1. download everything into data/raw/ (logs URL + SHA-256)
python python/run_prepare.py          # 2. build data/processed/ and run layout checks
python python/run_backtest.py --tests testA10 testA5   # 3a. development back-tests
#    -> commit methodology/methodology_v0.1.pdf before the held-out test
python python/run_backtest.py --tests testB5           # 3b. held-out test + interval coverage
python python/run_forecast.py         # 4. production forecast, 2030 and 2035
Rscript R/crosscheck.R                # 5. independent R engine agrees?
```

Step 1 prints a message for any file it cannot fetch automatically (for example
the HUD CHAS zip) with the landing page to download it from by hand.
Step 2 stops if a downloaded file does not match the layout the code expects.

See `USER_GUIDE.docx` for a step-by-step guide to every command in Python and R.

## Where things are

| Path | What it holds |
|---|---|
| `config/model.yaml` | Every assumption: shrinkage constants, vacancy targets, scenario definitions |
| `config/runs.yaml` | Launch date, calibration windows and targets for production and each back-test |
| `config/sources.yaml` | Every URL, API table and file layout code |
| `config/national_asfr.csv` | National fertility rates typed from NCHS reports, with citations |
| `config/counties_demo.csv` | The 16 demonstration counties in six states |
| `python/hnf/` | Python engine (one module per model component) |
| `R/hnf_engine.R` | R engine (same equations, written independently) |
| `data/data_dictionary.csv` | Every processed variable: source, table, transformation, missing values |
| `tests/` | Unit tests, parser tests on real-layout fixtures, end-to-end tests |
| `output/<run>/` | Results, diagnostics, run manifest (inputs' SHA-256, config, version) |

## Main outputs (`output/production/`)

- `summary.csv` — every county × scenario (low/base/high) × horizon (0, 5, 10) × measure:
  population, households, owners, renters, households by tenure and income band,
  required units (total and by band), stock, available units, gap, and the gap's
  decomposition (existing gap + growth in need − pipeline). Base-scenario population
  and households carry calibrated 80% prediction intervals (`epi_low`, `epi_high`).
- `demo_counties.csv` — the demonstration counties with scenarios side by side.
- `sensitivity.csv` — how much each assumption moves the 10-year gap.
- `diagnostics.csv` — every estimated parameter, raw and after shrinkage.
- `output/backtests/` — accuracy by measure, method, county size and growth class;
  model-vs-benchmark win rates; interval coverage.

## Citing

See `CITATION.cff`. Archive each release (for example on Zenodo) to get a DOI.

## License

MIT — see `LICENSE`.
