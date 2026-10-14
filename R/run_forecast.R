# Production forecast with the R engine.
#   Rscript R/run_forecast.R [data_dir] [out_dir]
# Defaults: data/processed and output/R. Run from the repository root.
# In RStudio: setwd() to the repository, optionally set data_dir / out_dir, then source() this file.

source("R/hnf_engine.R")
args <- commandArgs(trailingOnly = TRUE)
if (!exists("data_dir")) data_dir <- if (length(args) >= 1) args[1] else "data/processed"
if (!exists("out_dir")) out_dir <- if (length(args) >= 2) args[2] else "output/R"

cfg <- hnf_config(".")
res <- hnf_run("production", cfg, data_dir)
dir.create(file.path(out_dir, "production"), recursive = TRUE, showWarnings = FALSE)
fwrite(res$summary, file.path(out_dir, "production", "summary.csv"))

demo <- fread("config/counties_demo.csv", colClasses = c(geoid = "character"))
d <- res$summary[geoid %in% demo$geoid]
if (nrow(d)) fwrite(dcast(d, geoid + date + horizon + measure ~ scenario, value.var = "value"),
                    file.path(out_dir, "production", "demo_counties.csv"))
cat("R engine: projected", uniqueN(res$summary$geoid), "counties ->", file.path(out_dir, "production"), "\n")
