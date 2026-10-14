"""hnf: U.S. County Housing Needs Forecasting Model, Version 0.1.

Modules
-------
ages        age-group conventions shared by every module
io          configuration, processed-table I/O and the download provenance log
acquire     downloads every public source listed in config/sources.yaml
prepare     converts raw downloads into the processed tables both engines read
mortality   life tables -> five-year survival ratios
fertility   county fertility index
migration   cohort-residual net migration, shrinkage and scenario schedules
projection  cohort-component projection engine
households  headship and tenure
income      HUD income bands (CHAS shares)
supply      housing stock, permit realization factor and expected additions
gap         housing-unit requirement and gap
employment  optional employment-informed migration level (tested in back-test B5)
benchmarks  naive benchmark forecasts
validation  error metrics, empirical prediction intervals and coverage
pipeline    runs a forecast or back-test from a run specification
"""

__version__ = "0.1.0"
