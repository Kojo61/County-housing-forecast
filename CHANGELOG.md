# Changelog

All notable changes to the model are recorded here. Each released version is
tagged in git and archived with a DOI. Methodology changes cite the reviewer
comment or back-test result that motivated them.

## [0.1.0] — unreleased

### Added
- Cohort-component population engine (five-year steps, 18 age groups, two sexes).
- Cohort-residual net migration from two calibration windows with credibility
  shrinkage toward state rates; low/high migration from the observed windows.
- County fertility index (indirect standardization) with shrinkage.
- Headship and tenure by householder age, with empirical-Bayes shrinkage of ACS rates.
- Households by HUD income band from CHAS shares.
- Housing supply from census-to-census permit realization factors.
- Requirement, available supply, gap and its decomposition; composite scenarios and
  one-at-a-time sensitivity.
- Back-tests A10 (2010→2020), A5 (2010→2015) and B5 (2020→2025) with six benchmarks,
  error metrics by size and growth class, and empirical prediction intervals.
- Optional employment-informed migration level, evaluated out of sample in B5.
- Independent R engine and Python–R cross-check.
- Data acquisition with a provenance log and preparation with layout checks.

### Changed (set in development back-tests A10 and A5, October 2026)
- Migration shrinkage K lowered from 1,000 to 100. Tested 1,000, 250, 100, 25 and 0 on A10;
  K = 100 tied for lowest error (MAPE 7.02% -> 6.76%) and reduced upward bias (+4.84% -> +3.96%),
  which was largest in counties under 10,000 people. Confirmed on A5 (3.94% -> 3.84%).
- Housing pipeline uses the latest three permit years instead of five (production 2023-2025,
  B5 2017-2019, A10 2007-2009). On A10 the five-year window reached into the housing boom;
  housing-unit MAPE 7.84% -> 6.65%.
- B5 was not run before these choices were fixed.

### Held-out back-test B5 (run October 26, 2026, after methodology v0.1 was committed)
- Population 2020 -> 2025, 3,135 counties: model MAPE 3.19% (bias -1.53%);
  trend average 2.77%; no change 3.79%.
- Counties over 250,000 people: model 2.20%, the most accurate method
  (trend average 2.43%, no change 4.22%).
- 80% intervals calibrated on A5 covered 59% of outcomes. Production 5-year
  intervals use A5 and B5 errors together, as specified before B5 was run.
- Employment add-on: MAPE 3.15% vs 3.19%, but the employment-growth coefficient
  was -0.009 (s.e. 0.068), indistinguishable from zero, and accuracy fell for the
  largest and smallest counties. Kept out of production (use_in_production: false).

### Demonstration counties revised (October 26, 2026, after the production run)
- Whitman County, WA (53075) moved from the demonstration set to a stress-test
  appendix: its student-dominated population produced a large measured surplus
  (about 5,100 units against 17,000 households) that we could not explain.
- Replaced by Stevens County, WA (53065), a small rural county of similar size,
  chosen for its role before its results were viewed.
- The model, its settings and the national results are unchanged; Whitman's
  results remain in output/production/summary.csv and are reported in the appendix.