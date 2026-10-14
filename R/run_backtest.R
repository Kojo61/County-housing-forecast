# Back-tests with the R engine.
#   Rscript R/run_backtest.R [data_dir] [out_dir]
# Defaults: data/processed and output/R. Run from the repository root.
# In RStudio: setwd() to the repository, optionally set data_dir / out_dir, then source() this file.

source("R/hnf_engine.R")
args <- commandArgs(trailingOnly = TRUE)
if (!exists("data_dir")) data_dir <- if (length(args) >= 1) args[1] else "data/processed"
if (!exists("out_dir")) out_dir <- if (length(args) >= 2) args[2] else "output/R"

cfg <- hnf_config(".")
all_err <- list()
for (t in c("testA10", "testA5", "testB5")) {
  res <- tryCatch(hnf_run(t, cfg, data_dir), error = function(e) { message("skipped ", t, ": ", conditionMessage(e)); NULL })
  if (is.null(res)) next
  dir.create(file.path(out_dir, t), recursive = TRUE, showWarnings = FALSE)
  fwrite(res$summary, file.path(out_dir, t, "summary.csv"))
  fwrite(res$errors, file.path(out_dir, t, "errors.csv"))
  all_err[[t]] <- res$errors
}
e <- rbindlist(all_err)
m <- hnf_metrics(e)
dir.create(file.path(out_dir, "backtests"), recursive = TRUE, showWarnings = FALSE)
fwrite(m, file.path(out_dir, "backtests", "accuracy_summary.csv"))
print(m[method %in% c("model", "AVE", "CONST")], digits = 3)
