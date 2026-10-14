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
