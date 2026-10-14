# =============================================================================
# U.S. County Housing Needs Forecasting Model, Version 0.1: R engine
#
# An independent implementation of the model in R. It reads the same processed
# tables (data/processed/*.csv) and the same configuration (config/*.yaml) as
# the Python engine, and R/crosscheck.R verifies that both produce the same
# numbers. Two implementations that agree are a strong check that the code
# does what the methodology says.
#
# Conventions: population arrays are [county, sex, age] with sex 1 = male,
# 2 = female and age 1..18 = 0-4, 5-9, ..., 80-84, 85+.
# Requires: data.table, yaml
# =============================================================================

suppressPackageStartupMessages({
  library(data.table)
  library(yaml)
})

HH_GROUPS <- list("15-24" = 4:5, "25-34" = 6:7, "35-44" = 8:9, "45-54" = 10:11, "55-59" = 12,
                  "60-64" = 13, "65-74" = 14:15, "75-84" = 16:17, "85+" = 18)
HH_LABELS <- names(HH_GROUPS)
FERT_AGES <- 3:10            # 10-14 .. 45-49
TENURES <- c("owner", "renter")
LEVELS <- c("low", "base", "high")

# ---------------------------------------------------------------------------
# Configuration and tables
# ---------------------------------------------------------------------------
hnf_config <- function(root = ".") {
  list(model = read_yaml(file.path(root, "config", "model.yaml"), fileEncoding = "UTF-8"),
       runs = read_yaml(file.path(root, "config", "runs.yaml"), fileEncoding = "UTF-8"),
       root = root)
}

read_tbl <- function(dir, name, required = TRUE) {
  path <- file.path(dir, paste0(name, ".csv"))
  if (!file.exists(path)) {
    if (required) stop("Processed table '", name, "' not found at ", path)
    return(NULL)
  }
  text_cols <- c("geoid", "state", "geo", "series", "date", "sex", "source", "table_id",
                 "hh_age", "tenure", "band", "measure")
  hdr <- names(fread(path, nrows = 0))
  cc <- setNames(rep("character", length(intersect(text_cols, hdr))), intersect(text_cols, hdr))
  fread(path, colClasses = cc)
}

years_between <- function(a, b) as.numeric(as.Date(b) - as.Date(a)) / 365.25

add_years <- function(date, y) {
  d <- as.POSIXlt(as.Date(date))
  d$year <- d$year + y
  format(as.Date(d), "%Y-%m-%d")
}

pop_array <- function(pop, s, dt, geoids) {
  d <- pop[series == s & date == dt]
  if (nrow(d) == 0) stop("No population for series=", s, " date=", dt)
  arr <- array(NA_real_, dim = c(length(geoids), 2, 18))
  ci <- match(d$geoid, geoids)
  ok <- !is.na(ci)
  arr[cbind(ci[ok], ifelse(d$sex[ok] == "F", 2L, 1L), d$age[ok] + 1L)] <- d$pop[ok]
  arr
}

nearest_date <- function(pop, s, target) {
  dates <- sort(unique(pop[series == s, date]))
  dates[which.min(abs(as.numeric(as.Date(dates) - as.Date(target))))]
}

# ---------------------------------------------------------------------------
# Shrinkage
# ---------------------------------------------------------------------------
group_ratio <- function(num, den, groups) {
  dims <- dim(num)
  C <- length(groups)
  N <- matrix(num, nrow = C); D <- matrix(den, nrow = C)
  N[is.na(N)] <- 0; D[is.na(D)] <- 0
  out <- matrix(NA_real_, nrow = C, ncol = ncol(N))
  for (g in unique(groups)) {
    idx <- which(groups == g)
    n <- colSums(N[idx, , drop = FALSE]); d <- colSums(D[idx, , drop = FALSE])
    r <- ifelse(d > 0, n / d, NA_real_)
    out[idx, ] <- matrix(r, nrow = length(idx), ncol = length(r), byrow = TRUE)
  }
  if (is.null(dims)) as.vector(out) else array(out, dim = dims)
}

credibility <- function(rate, exposure, group_rate, K) {
  n <- exposure; n[is.na(n)] <- 0; n <- pmax(n, 0)
  w <- if (K > 0) n / (n + K) else (n > 0) * 1
  r0 <- rate; r0[!is.finite(r0)] <- 0
  shrunk <- ifelse(is.finite(rate), w * r0 + (1 - w) * group_rate, group_rate)
  if (!is.null(dim(rate))) dim(shrunk) <- dim(rate)
  list(value = shrunk, w = w)
}

empirical_bayes <- function(est, se, target) {
  se2 <- se^2
  dev <- est - target
  ok <- is.finite(dev) & is.finite(se2)
  tau2 <- numeric(ncol(est))
  for (j in seq_len(ncol(est))) {
    m <- ok[, j]
    if (sum(m) > 2) {
      x <- dev[m, j]
      tau2[j] <- max(0, mean((x - mean(x))^2) - mean(se2[m, j]))   # population variance
    }
  }
  T2 <- matrix(tau2, nrow = nrow(est), ncol = ncol(est), byrow = TRUE)
  w <- ifelse(!is.na(se2) & se2 > 0, T2 / (T2 + se2), 1)
  w[!is.finite(w)] <- 0
  e0 <- est; e0[!is.finite(e0)] <- 0
  shrunk <- ifelse(is.finite(est), w * e0 + (1 - w) * target, target)
  dim(shrunk) <- dim(est); dim(w) <- dim(est)
  list(value = shrunk, w = w)
}

# ---------------------------------------------------------------------------
# Mortality
# ---------------------------------------------------------------------------
survival_ratios <- function(L, l0) {
  stopifnot(length(L) == 101)
  T <- rev(cumsum(rev(L)))                                   # T[x + 1] = T_x
  L5 <- sapply(0:17, function(g) sum(L[(5 * g + 1):(5 * g + 5)]))
  S <- numeric(18)
  S[1:16] <- L5[2:17] / L5[1:16]
  S[17] <- L5[18] / L5[17]
  S[18] <- T[91] / T[86]
  list(S = S, Sb = L5[1] / (5 * l0))
}

survival_lookup <- function(life) {
  out <- list()
  for (key in unique(life[, paste(table_id, geo, sex, sep = "|")])) {
    p <- strsplit(key, "|", fixed = TRUE)[[1]]
    g <- life[table_id == p[1] & geo == p[2] & sex == p[3]][order(age)]
    out[[key]] <- survival_ratios(g$Lx, g$lx[1])
  }
  out
}

county_survival <- function(states, table_id, lookup) {
  C <- length(states)
  S <- array(NA_real_, dim = c(C, 2, 18)); Sb <- matrix(NA_real_, C, 2)
  for (i in seq_len(C)) for (s in 1:2) {
    sex <- c("M", "F")[s]
    key <- paste(table_id, states[i], sex, sep = "|")
    if (is.null(lookup[[key]])) key <- paste(table_id, "US", sex, sep = "|")
    if (is.null(lookup[[key]])) stop("Life table ", table_id, " missing for ", states[i], " ", sex)
    S[i, s, ] <- lookup[[key]]$S; Sb[i, s] <- lookup[[key]]$Sb
  }
  list(S = S, Sb = Sb)
}

# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------
safe_div <- function(a, b) { r <- ifelse(b > 0, a / b, NA_real_); if (!is.null(dim(a))) dim(r) <- dim(a); r }

window_rates <- function(Ps, Pe, S, Sb, years, states, K, beta, births = NULL) {
  C <- dim(Ps)[1]
  NM <- array(0, dim = c(C, 2, 18))
  NM[, , 1:16] <- Pe[, , 2:17, drop = FALSE] - Ps[, , 1:16, drop = FALSE] * S[, , 1:16, drop = FALSE]
  nm85 <- Pe[, , 18] - (Ps[, , 17] * S[, , 17] + Ps[, , 18] * S[, , 18])
  NM[, , 17] <- nm85; NM[, , 18] <- nm85
  expo <- Ps
  e85 <- Ps[, , 17] + Ps[, , 18]
  expo[, , 17] <- e85; expo[, , 18] <- e85
  scale <- 5 / years
  NM <- NM * scale
  m_raw <- safe_div(NM, expo)
  m_state <- group_ratio(NM, expo, states)
  cr <- credibility(m_raw, expo, m_state, K)
  if (!is.null(births)) {
    NMb <- (Pe[, , 1] - births * Sb) * scale
    mb_raw <- safe_div(NMb, births)
    mbs <- group_ratio(NMb, births, states)
    mb <- credibility(mb_raw, births, mbs, K)$value
  } else {
    mb_raw <- beta * m_raw[, , 1]
    mb <- beta * cr$value[, , 1]
  }
  mb_raw <- matrix(mb_raw, nrow = C); mb <- matrix(mb, nrow = C)
  nm5 <- apply(NM[, , 1:17, drop = FALSE], 1, sum)
  crude <- ifelse(apply(Ps, 1, sum) > 0, nm5 / apply(Ps, 1, sum), NA_real_)
  list(m_raw = m_raw, m = cr$value, w = array(cr$w, dim = dim(NM)), m_state = m_state,
       mb_raw = mb_raw, mb = mb, crude = crude)
}

implied_net_migrants <- function(P, m) {
  apply(P[, , 1:16, drop = FALSE] * m[, , 1:16, drop = FALSE], 1, sum) +
    rowSums(matrix((P[, , 17] + P[, , 18]) * m[, , 17], nrow = dim(P)[1]))
}

scenario_schedules <- function(wins, weights, P0) {
  wts <- weights / sum(weights)
  base_m <- Reduce(`+`, Map(function(w, x) x$m * w, wts, wins))
  base_mb <- Reduce(`+`, Map(function(w, x) x$mb * w, wts, wins))
  implied <- sapply(wins, function(x) implied_net_migrants(P0, x$m))
  if (is.null(dim(implied))) implied <- matrix(implied, nrow = 1)
  lo <- apply(implied, 1, which.min); hi <- apply(implied, 1, which.max)
  pick <- function(idx) {
    m <- base_m; mb <- base_mb
    for (i in seq_along(idx)) { m[i, , ] <- wins[[idx[i]]]$m[i, , ]; mb[i, ] <- wins[[idx[i]]]$mb[i, ] }
    list(m = m, mb = mb)
  }
  list(base = list(m = base_m, mb = base_mb), low = pick(lo), high = pick(hi))
}

shift_schedule <- function(sch, P0, m_state, delta) {
  profile <- abs(m_state); profile[!is.finite(profile)] <- 0
  weight <- implied_net_migrants(P0, profile)
  pop <- apply(P0, 1, sum)
  scale <- ifelse(weight > 0, delta * pop / weight, 0)
  list(m = sch$m + profile * array(scale, dim = dim(profile)),
       mb = sch$mb + profile[, , 1] * scale)
}

# ---------------------------------------------------------------------------
# Fertility and projection
# ---------------------------------------------------------------------------
fertility_index <- function(births, expected, states, K) {
  ok <- is.finite(births) & is.finite(expected) & expected > 0
  national <- sum(births[ok]) / sum(expected[ok])
  k_raw <- ifelse(ok, births / expected / national, NA_real_)
  k_state <- group_ratio(ifelse(ok, births, 0), ifelse(ok, expected, 0), states) / national
  cr <- credibility(k_raw, ifelse(ok, expected, 0), k_state, K)
  k <- cr$value; k[!is.finite(k)] <- 1
  list(k = k, k_raw = k_raw, w = cr$w)
}

project_population <- function(P0, S, Sb, m, mb, asfr, k, steps, srb = 1.05, floor = 0) {
  fac <- S + m; fac[fac < floor] <- floor
  facb <- Sb + mb; facb[facb < floor] <- floor
  C <- dim(P0)[1]
  share <- c(srb / (1 + srb), 1 / (1 + srb))
  pops <- list(P0); P <- P0
  for (s in seq_len(steps)) {
    Pn <- array(0, dim = dim(P))
    Pn[, , 2:17] <- P[, , 1:16, drop = FALSE] * fac[, , 1:16, drop = FALSE]
    Pn[, , 18] <- P[, , 17] * fac[, , 17] + P[, , 18] * fac[, , 18]
    W <- matrix(0.5 * (P[, 2, FERT_AGES, drop = FALSE] + Pn[, 2, FERT_AGES, drop = FALSE]), nrow = C)
    B <- 5 * k * as.vector(W %*% asfr)
    Pn[, , 1] <- cbind(B * share[1], B * share[2]) * facb
    pops[[s + 1]] <- Pn; P <- Pn
  }
  pops
}

# ---------------------------------------------------------------------------
# Households, income bands, supply and gap
# ---------------------------------------------------------------------------
to_hh_groups <- function(P) {           # P: [C, 2, 18] -> C x 9
  both <- matrix(P[, 1, ] + P[, 2, ], nrow = dim(P)[1])
  sapply(HH_GROUPS, function(a) rowSums(both[, a, drop = FALSE]))
}

hh_matrix <- function(hb, geoids, col) {
  out <- matrix(NA_real_, length(geoids), 9)
  if (!(col %in% names(hb))) return(out)
  ci <- match(hb$geoid, geoids); ai <- match(hb$hh_age, HH_LABELS)
  ok <- !is.na(ci) & !is.na(ai)
  out[cbind(ci[ok], ai[ok])] <- hb[[col]][ok]
  out
}

prop_se <- function(p, n, se_x, se_n) {
  rad <- se_x^2 - p^2 * se_n^2
  rad <- ifelse(!is.na(rad) & rad < 0, se_x^2 + p^2 * se_n^2, rad)
  r <- ifelse(n > 0, sqrt(rad) / n, NA_real_)
  dim(r) <- dim(p); r
}

hh_rates <- function(hb, geoids, states, shrink = TRUE) {
  pop <- hh_matrix(hb, geoids, "pop"); own <- hh_matrix(hb, geoids, "owners")
  ren <- hh_matrix(hb, geoids, "renters"); hh <- own + ren
  clip01 <- function(x) { x[!is.na(x) & x < 0] <- 0; x[!is.na(x) & x > 1] <- 1; x }
  hr_raw <- clip01(ifelse(pop > 0, hh / pop, NA_real_)); dim(hr_raw) <- dim(pop)
  own_raw <- clip01(ifelse(hh > 0, own / hh, NA_real_)); dim(own_raw) <- dim(pop)
  fin <- is.finite(hr_raw); fin2 <- is.finite(own_raw)
  hr_t <- group_ratio(ifelse(fin, hh, 0), ifelse(fin, pop, 0), states); dim(hr_t) <- dim(pop)
  own_t <- group_ratio(ifelse(fin2, own, 0), ifelse(fin2, hh, 0), states); dim(own_t) <- dim(pop)
  hr <- hr_raw; ow <- own_raw
  if (all(c("pop_se", "owners_se", "renters_se") %in% names(hb)) && shrink) {
    z <- function(x) { x[is.na(x)] <- 0; x }
    se_p <- z(hh_matrix(hb, geoids, "pop_se")); se_o <- z(hh_matrix(hb, geoids, "owners_se"))
    se_r <- z(hh_matrix(hb, geoids, "renters_se")); se_h <- sqrt(se_o^2 + se_r^2)
    hr <- empirical_bayes(hr_raw, prop_se(hr_raw, pop, se_h, se_p), hr_t)$value
    ow <- empirical_bayes(own_raw, prop_se(own_raw, hh, se_o, se_h), own_t)$value
  }
  hr <- ifelse(is.finite(hr), hr, hr_t); dim(hr) <- dim(pop)
  ow <- ifelse(is.finite(ow), ow, own_t); dim(ow) <- dim(pop)
  list(hr = hr, own = ow)
}

project_households <- function(P, hr, own) {
  hh <- to_hh_groups(P) * hr
  o <- hh * own
  list(households = rowSums(hh), owners = rowSums(o), renters = rowSums(hh - o))
}

chas_shares <- function(chas, geoids, bands) {
  counts <- array(NA_real_, dim = c(length(geoids), 2, length(bands)))
  ci <- match(chas$geoid, geoids); ti <- match(chas$tenure, TENURES); bi <- match(chas$band, bands)
  ok <- !is.na(ci) & !is.na(ti) & !is.na(bi)
  counts[cbind(ci[ok], ti[ok], bi[ok])] <- chas$households[ok]
  tot <- apply(counts, c(1, 2), sum)
  out <- counts
  for (b in seq_along(bands)) out[, , b] <- ifelse(tot > 0, counts[, , b] / tot, NA_real_)
  out
}

series_value <- function(df, geoids, s, value, dt = NULL) {
  d <- df[series == s]
  if (!is.null(dt)) d <- d[date == dt]
  agg <- d[, .(v = sum(get(value))), by = geoid]
  agg$v[match(geoids, agg$geoid)]
}

permits_by_year <- function(permits, geoids, y0, y1) {
  yrs <- y0:y1
  missing <- setdiff(yrs, unique(permits$year))
  if (length(missing)) stop("Permit years missing: ", paste(missing, collapse = ", "))
  out <- matrix(0, length(geoids), length(yrs))
  d <- permits[year >= y0 & year <= y1, .(units = sum(units)), by = .(geoid, year)]
  ci <- match(d$geoid, geoids); ok <- !is.na(ci)
  out[cbind(ci[ok], d$year[ok] - y0 + 1)] <- d$units[ok]
  out
}

realization_factor <- function(hu_s, hu_e, permits, states, K, bounds) {
  change <- hu_e - hu_s
  ok <- is.finite(change) & is.finite(permits)
  rf_raw <- ifelse(ok & permits > 0, change / permits, NA_real_)
  rf_state <- group_ratio(ifelse(ok, change, 0), ifelse(ok, permits, 0), states)
  rf <- credibility(rf_raw, ifelse(ok, permits, 0), rf_state, K)$value
  pmin(pmax(rf, bounds[1]), bounds[2])
}

# ---------------------------------------------------------------------------
# The run (mirrors python/hnf/pipeline.py)
# ---------------------------------------------------------------------------
births_window <- function(births, geoids, s, y0, y1) {
  d <- births[series == s & year_end >= y0 & year_end <= y1,
              .(b = sum(births), n = uniqueN(year_end)), by = geoid]
  i <- match(geoids, d$geoid)
  ifelse(!is.na(i) & d$n[i] == (y1 - y0 + 1), d$b[i], NA_real_)
}

hnf_run <- function(name, cfg, data_dir) {
  spec <- cfg$runs[[name]]; model <- cfg$model
  pop <- read_tbl(data_dir, "population"); life <- read_tbl(data_dir, "life_tables")
  asfr_tab <- read_tbl(data_dir, "asfr")
  lookup <- survival_lookup(life)
  srb <- model$population$sex_ratio_at_birth
  launch <- spec$launch; steps <- as.integer(spec$steps)

  geoids <- sort(unique(pop[series == launch$series & date == launch$date, geoid]))
  geoids <- geoids[!(substr(geoids, 1, 2) %in% model$geography$exclude_states)]
  P0 <- pop_array(pop, launch$series, launch$date, geoids)
  keep <- apply(is.finite(P0), 1, all) & apply(P0, 1, sum, na.rm = TRUE) > 0
  wraw <- list()
  for (w in spec$migration_windows) {
    Ps <- pop_array(pop, w$start$series, w$start$date, geoids)
    Pe <- pop_array(pop, w$end$series, w$end$date, geoids)
    keep <- keep & apply(is.finite(Ps), 1, all) & apply(is.finite(Pe), 1, all)
    b <- NULL
    if (!is.null(w$births)) {
      bt <- births_window(read_tbl(data_dir, "births"), geoids, w$births$series, w$births$years[1], w$births$years[2])
      keep <- keep & is.finite(bt)
      b <- cbind(bt * srb / (1 + srb), bt / (1 + srb))
    }
    wraw[[w$id]] <- list(w = w, Ps = Ps, Pe = Pe, b = b)
  }
  geoids <- geoids[keep]; P0 <- P0[keep, , , drop = FALSE]
  states <- substr(geoids, 1, 2)

  K <- model$migration$shrinkage_K; beta <- model$population$births_migration_factor
  wins <- list()
  for (id in names(wraw)) {
    x <- wraw[[id]]; sv <- county_survival(states, x$w$life_table, lookup)
    wins[[id]] <- window_rates(x$Ps[keep, , , drop = FALSE], x$Pe[keep, , , drop = FALSE], sv$S, sv$Sb,
                               years_between(x$w$start$date, x$w$end$date), states, K, beta,
                               if (is.null(x$b)) NULL else x$b[keep, , drop = FALSE])
  }
  ids <- vapply(spec$migration_windows, function(w) w$id, "")
  weights <- vapply(ids, function(i) if (is.null(spec$migration_weights[[i]])) 1 else spec$migration_weights[[i]], 0)
  sched <- scenario_schedules(wins[ids], weights, P0)
  m_state_base <- Reduce(`+`, Map(function(w, i) wins[[i]]$m_state * w, weights / sum(weights), ids))

  if (!is.null(spec$employment) && file.exists(file.path(data_dir, "employment.csv"))) {
    emp <- read_tbl(data_dir, "employment")
    growth <- function(y0, y1, col) {
      a <- emp[year == y0][match(geoids, geoid)][[col]]; b <- emp[year == y1][match(geoids, geoid)][[col]]
      g <- log(b / a) / (y1 - y0); g[!is.finite(g)] <- NA; g
    }
    feats <- function(y, prior) data.frame(emp_growth = growth(y[1], y[2], "emp"),
                                           wage_growth = growth(y[1], y[2], "wkly_wage"),
                                           prior_mig = wins[[prior]]$crude,
                                           state = factor(states, levels = sort(unique(states))))
    tr <- spec$employment$train; ap <- spec$employment$apply
    dtrain <- feats(tr$emp_years, tr$prior_window); dtrain$y <- wins[[tr$target_window]]$crude
    f <- if (length(unique(states)) > 1) y ~ emp_growth + wage_growth + prior_mig + state else
      y ~ emp_growth + wage_growth + prior_mig
    fit <- lm(f, data = dtrain)
    pred <- predict(fit, newdata = feats(ap$emp_years, ap$prior_window))
    base_crude <- implied_net_migrants(P0, sched$base$m) / apply(P0, 1, sum)
    delta <- ifelse(is.finite(pred), pred - base_crude, 0)
    if (name != "production" || isTRUE(model$employment$use_in_production))
      sched$econ <- shift_schedule(sched$base, P0, m_state_base, delta)
  }

  fz <- spec$fertility
  a <- asfr_tab[year == fz$asfr_year][order(age)]
  asfr <- a$births_per_1000_women / 1000
  fi <- fz$index
  b_obs <- births_window(read_tbl(data_dir, "births"), geoids, fi$births_series, fi$years[1], fi$years[2])
  expected <- 0
  for (y in fi$years[1]:fi$years[2]) {
    W <- pop_array(pop, fi$pop_series, nearest_date(pop, fi$pop_series, sprintf("%d-07-01", y - 1)), geoids)
    expected <- expected + as.vector(matrix(W[, 2, FERT_AGES], nrow = length(geoids)) %*% asfr)
  }
  fert <- fertility_index(b_obs, expected, states, model$fertility$shrinkage_K)

  sv <- county_survival(states, spec$projection_life_table, lookup)
  floor <- model$migration$min_survival_plus_migration
  mig_levels <- c(LEVELS, if (!is.null(sched$econ)) "econ")
  pops <- lapply(setNames(mig_levels, mig_levels), function(l)
    project_population(P0, sv$S, sv$Sb, sched[[l]]$m, sched[[l]]$mb, asfr, fert$k, steps, srb, floor))
  dates <- vapply(0:steps, function(s) add_years(launch$date, 5 * s), "")

  hh <- NULL
  if (!is.null(spec$households)) {
    hb <- read_tbl(data_dir, "households_base")
    shrink <- isTRUE(model$households$shrink_acs_rates)
    base <- hh_rates(hb[series == spec$households$base], geoids, states, shrink)
    alts <- lapply(spec$households$alternatives, function(s) hh_rates(hb[series == s], geoids, states, shrink)$hr)
    stack <- c(list(base$hr), alts)
    hr_sc <- list(base = base$hr,
                  low = Reduce(function(x, y) pmin(x, y, na.rm = TRUE), stack),
                  high = Reduce(function(x, y) pmax(x, y, na.rm = TRUE), stack))
    hh <- list()
    for (ml in mig_levels) for (hl in LEVELS)
      hh[[paste(ml, hl)]] <- lapply(0:steps, function(s)
        project_households(pops[[ml]][[s + 1]], base$hr + (s / steps) * (hr_sc[[hl]] - base$hr), base$own))
  }
  shares <- NULL
  bands <- unlist(model$income_bands)
  if (!is.null(spec$chas)) {
    ch <- read_tbl(data_dir, "chas"); shares <- chas_shares(ch[source == spec$chas], geoids, bands)
  }
  sup <- NULL
  if (!is.null(spec$supply)) {
    sp <- spec$supply; hu <- read_tbl(data_dir, "housing_units"); pm <- read_tbl(data_dir, "permits")
    rz <- sp$realization
    hs <- series_value(hu, geoids, rz$start$series, "hu", rz$start$date)
    he <- series_value(hu, geoids, rz$end$series, "hu", rz$end$date)
    p_rz <- rowSums(permits_by_year(pm, geoids, rz$permit_years[1], rz$permit_years[2]))
    rf <- realization_factor(hs, he, p_rz, states, model$supply$shrinkage_K, unlist(model$supply$bounds))
    trend <- (he - hs) / (rz$permit_years[2] - rz$permit_years[1] + 1)
    pp <- permits_by_year(pm, geoids, sp$projection_permit_years[1], sp$projection_permit_years[2])
    annual <- list(low = apply(pp, 1, min), base = rowMeans(pp), high = apply(pp, 1, max))
    hu0 <- series_value(hu, geoids, sp$base_hu$series, "hu", sp$base_hu$date)
    additions <- function(level, yrs) ifelse(p_rz > 0, rf * annual[[level]] * yrs, trend * yrs)
    sup <- list(rf = rf, hu0 = hu0, additions = additions, p_rz = p_rz, trend = trend)
    if (!is.null(sp$conditional_permit_years))
      sup$cond <- rowMeans(permits_by_year(pm, geoids, sp$conditional_permit_years[1], sp$conditional_permit_years[2]))
  }
  avail <- NULL
  if (!is.null(spec$vacancy)) {
    v <- read_tbl(data_dir, "vacancy")[source == spec$vacancy]
    v <- v[match(geoids, v$geoid)]
    oa <- model$vacancy$other_vacant_available_share
    avail <- ifelse(v$hu_total > 0, 1 - (v$seasonal + v$migrant + (1 - oa) * v$other_vacant) / v$hu_total, NA_real_)
    avail <- pmin(pmax(avail, 0), 1)
  }

  vac <- model$vacancy
  block <- function(ch) {
    out <- list(population = lapply(pops[[ch$migration]], function(p) apply(p, 1, sum)))
    if (!is.null(hh)) {
      hs <- hh[[paste(ch$migration, ch$headship)]]
      out$households <- lapply(hs, `[[`, "households")
      out$owners <- lapply(hs, `[[`, "owners"); out$renters <- lapply(hs, `[[`, "renters")
      vo <- vac$target_owner[[ch$vacancy]]; vr <- vac$target_renter[[ch$vacancy]]
      req <- Map(function(o, r) o / (1 - vo) + r / (1 - vr), out$owners, out$renters)
      if (!is.null(shares)) {
        for (ti in 1:2) for (bi in seq_along(bands)) {
          tot <- if (ti == 1) out$owners else out$renters
          out[[paste0(TENURES[ti], "_", bands[bi])]] <- lapply(tot, function(x) x * shares[, ti, bi])
        }
        for (b in bands) out[[paste0("hh_", b)]] <- Map(`+`, out[[paste0("owner_", b)]], out[[paste0("renter_", b)]])
        out$hh_gt80 <- Map(`+`, out$hh_80_100, out$hh_gt100)
        for (b in bands) out[[paste0("required_units_", b)]] <-
          Map(function(o, r) o / (1 - vo) + r / (1 - vr), out[[paste0("owner_", b)]], out[[paste0("renter_", b)]])
      }
      if (!is.null(sup) && !is.null(avail)) {
        for (s in 0:steps) {
          add <- sup$additions(ch$supply, 5 * s)
          a0 <- sup$hu0 * avail; pipe <- add * avail
          comp <- list(required_units = req[[s + 1]], stock = sup$hu0 + add, available_units = a0 + pipe,
                       gap = req[[s + 1]] - (a0 + pipe), existing_gap = req[[1]] - a0,
                       growth_in_need = req[[s + 1]] - req[[1]], pipeline = pipe)
          for (k in names(comp)) out[[k]][[s + 1]] <- comp[[k]]
        }
      }
    } else if (!is.null(sup)) {
      out$stock <- lapply(0:steps, function(s) sup$hu0 + sup$additions(ch$supply, 5 * s))
    }
    out
  }
  comp <- model$scenarios$composite
  blocks <- lapply(comp, block)
  if ("econ" %in% mig_levels)
    blocks$econ <- block(list(migration = "econ", headship = "base", vacancy = "base", supply = "base"))
  summary <- rbindlist(lapply(names(blocks), function(sc) rbindlist(lapply(names(blocks[[sc]]), function(ms)
    rbindlist(lapply(0:steps, function(s) data.table(run = name, geoid = geoids, date = dates[s + 1],
      horizon = 5 * s, scenario = sc, measure = ms, value = blocks[[sc]][[ms]][[s + 1]])))))))

  res <- list(summary = summary)
  if (!is.null(spec$targets)) res$errors <- hnf_score(spec, name, data_dir, pop, geoids, states, P0, blocks, sup, dates)
  res
}

hnf_score <- function(spec, name, data_dir, pop, geoids, states, P0, blocks, sup, dates) {
  h <- spec$history
  pl <- apply(pop_array(pop, h[[1]]$series, h[[1]]$date, geoids), 1, sum)
  ps <- apply(pop_array(pop, h[[2]]$series, h[[2]]$date, geoids), 1, sum)
  p0 <- apply(P0, 1, sum)
  yl <- years_between(h[[1]]$date, spec$launch$date); ys <- years_between(h[[2]]$date, spec$launch$date)
  out <- list()
  add <- function(measure, method, f, a, hz) out[[length(out) + 1]] <<-
    data.table(run = name, geoid = geoids, measure = measure, method = method, horizon = hz, forecast = f, actual = a)
  gsum <- function(x) { s <- tapply(x, states, sum, na.rm = TRUE); as.vector(s[states]) }
  for (t in spec$targets) {
    st <- as.integer(t$step); hz <- 5 * st
    sfx <- if (isTRUE(t$diagnostic)) "_diagnostic" else ""
    if (t$source == "population") {
      a <- apply(pop_array(pop, t$series, t$date, geoids), 1, sum)
      add(t$measure, paste0("model", sfx), blocks$base$population[[st + 1]], a, hz)
      if (!is.null(blocks$econ)) add(t$measure, "model_econ", blocks$econ$population[[st + 1]], a, hz)
      lin <- pmax(p0 + hz * (p0 - pl) / yl, 0); ex <- p0 * (p0 / pl)^(hz / yl)
      sl <- gsum(p0); stl <- gsum(pl); shr <- p0 / sl * (sl + hz * (sl - stl) / yl)
      add(t$measure, "CONST", p0, a, hz); add(t$measure, "LIN", lin, a, hz); add(t$measure, "EXP", ex, a, hz)
      add(t$measure, "EXP_RECENT", p0 * (p0 / ps)^(hz / ys), a, hz); add(t$measure, "SHR", shr, a, hz)
      add(t$measure, "AVE", (lin + ex + shr) / 3, a, hz)
    } else if (t$source == "households_base") {
      hb <- read_tbl(data_dir, "households_base")[series == t$series]
      hb[, households := owners + renters]
      agg <- hb[, .(v = sum(get(t$measure))), by = geoid]
      a <- agg$v[match(geoids, agg$geoid)]
      add(t$measure, "model", blocks$base[[t$measure]][[st + 1]], a, hz)
      add(t$measure, "CONST_RATIO", blocks$base[[t$measure]][[1]] / p0 * blocks$base$population[[st + 1]], a, hz)
    } else if (t$source == "housing_units") {
      hu <- read_tbl(data_dir, "housing_units")
      a <- series_value(hu, geoids, t$series, "hu", t$date)
      add(t$measure, paste0("model", sfx), blocks$base$stock[[st + 1]], a, hz)
      if (!is.null(sup)) {
        add(t$measure, "CONST", sup$hu0, a, hz)
        add(t$measure, "LIN", pmax(sup$hu0 + sup$trend * hz, 0), a, hz)
        if (!is.null(sup$cond))
          add(t$measure, "model_conditional_permits",
              sup$hu0 + ifelse(sup$p_rz > 0, sup$rf * sup$cond * hz, sup$trend * hz), a, hz)
      }
    }
  }
  e <- rbindlist(out)
  e[, `:=`(error = forecast - actual, pct_error = ifelse(actual != 0, 100 * (forecast - actual) / actual, NA_real_))]
  e
}

hnf_metrics <- function(e, by = c("run", "measure", "method")) {
  e[is.finite(pct_error), .(n = .N, MAE = mean(abs(error)), RMSE = sqrt(mean(error^2)),
                            MAPE = mean(abs(pct_error)), MedAPE = median(abs(pct_error)),
                            MALPE = mean(pct_error)), by = by]
}
