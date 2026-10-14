# Cross-check: do the R and Python engines produce the same numbers?
#   Rscript R/crosscheck.R [data_dir] [python_output_dir]
# Run the Python back-tests and forecast on the same data first. Exits with
# status 1 if any value differs by more than 1e-6 (relative).
# In RStudio: setwd() to the repository, optionally set data_dir / py_out, then source() this file.

source("R/hnf_engine.R")
args <- commandArgs(trailingOnly = TRUE)
if (!exists("data_dir")) data_dir <- if (length(args) >= 1) args[1] else "data/processed"
if (!exists("py_out")) py_out <- if (length(args) >= 2) args[2] else "output"
cfg <- hnf_config(".")
tol <- 1e-6
rows <- list()

for (run in c("production", "testA10", "testA5", "testB5")) {
  py_file <- file.path(py_out, run, "summary.csv")
  if (!file.exists(py_file)) { message("no Python output for ", run, "; skipped"); next }
  r <- hnf_run(run, cfg, data_dir)
  py <- fread(py_file, colClasses = c(geoid = "character", date = "character"))
  m <- merge(r$summary, py[, .(geoid, date, horizon, scenario, measure, value_py = value)],
             by = c("geoid", "date", "horizon", "scenario", "measure"), all = TRUE)
  rel <- abs(m$value - m$value_py) / pmax(abs(m$value_py), 1)
  rows[[length(rows) + 1]] <- data.table(run = run, table = "summary", compared = sum(!is.na(rel)),
                                         only_in_one = sum(is.na(m$value) | is.na(m$value_py)),
                                         max_rel_diff = max(rel, na.rm = TRUE))
  if (!is.null(r$errors)) {
    pe <- fread(file.path(py_out, run, "errors.csv"), colClasses = c(geoid = "character"))
    me <- merge(r$errors, pe[, .(geoid, measure, method, f_py = forecast)], by = c("geoid", "measure", "method"))
    rel_e <- abs(me$forecast - me$f_py) / pmax(abs(me$f_py), 1)
    rows[[length(rows) + 1]] <- data.table(run = run, table = "errors", compared = sum(!is.na(rel_e)),
                                           only_in_one = nrow(r$errors) - nrow(me),
                                           max_rel_diff = max(rel_e, na.rm = TRUE))
  }
}
report <- rbindlist(rows)
report[, agree := max_rel_diff < tol & only_in_one == 0]
print(report)
dir.create(file.path(py_out, "crosscheck"), recursive = TRUE, showWarnings = FALSE)
fwrite(report, file.path(py_out, "crosscheck", "r_vs_python.csv"))
if (!all(report$agree)) {
  if (interactive()) stop("R and Python disagree; see the table above.") else quit(status = 1)
}
cat("R and Python engines agree on every compared value.\n")
