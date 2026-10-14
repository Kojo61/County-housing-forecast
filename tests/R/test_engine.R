# Unit tests for the R engine.   Rscript tests/R/test_engine.R   (from the repository root)
library(testthat)
source("R/hnf_engine.R")

test_that("survival ratios are exact for a constant force of mortality", {
  mu <- 0.02; x <- 0:100
  l <- 100000 * exp(-mu * x)
  L <- l * (1 - exp(-mu)) / mu; L[101] <- l[101] / mu
  s <- survival_ratios(L, l[1])
  expect_equal(s$S, rep(exp(-5 * mu), 18), tolerance = 1e-12)
  expect_equal(s$Sb, (1 - exp(-5 * mu)) / (5 * mu), tolerance = 1e-12)
})

test_that("cohort residual recovers the rates used to project", {
  set.seed(1); C <- 6
  P0 <- array(runif(C * 36, 100, 5000), dim = c(C, 2, 18))
  S <- array(runif(C * 36, 0.8, 0.999), dim = c(C, 2, 18))
  Sb <- matrix(runif(C * 2, 0.97, 0.995), C)
  m <- array(rnorm(C * 36, 0, 0.05), dim = c(C, 2, 18)); m[, , 18] <- m[, , 17]
  mb <- matrix(rnorm(C * 2, 0, 0.03), C)
  asfr <- c(0.0002, 0.013, 0.056, 0.09, 0.094, 0.054, 0.013, 0.001)
  P1 <- project_population(P0, S, Sb, m, mb, asfr, rep(1, C), 1)[[2]]
  W <- matrix(0.5 * (P0[, 2, FERT_AGES] + P1[, 2, FERT_AGES]), nrow = C)
  B <- 5 * as.vector(W %*% asfr)
  births <- cbind(B * 1.05 / 2.05, B / 2.05)
  est <- window_rates(P0, P1, S, Sb, 5, rep("55", C), K = 0, beta = 0.5, births = births)
  expect_equal(est$m_raw, m, tolerance = 1e-10)
  expect_equal(est$mb_raw, mb, tolerance = 1e-10)
})

test_that("credibility weights and group ratios behave as documented", {
  cr <- credibility(c(0.1, 0.1), c(1000, 0), c(0, 0), 1000)
  expect_equal(cr$value, c(0.05, 0))
  expect_equal(group_ratio(c(1, 3, 5), c(10, 10, 10), c("a", "a", "b")), c(0.2, 0.2, 0.5))
})

test_that("gap decomposition is an identity", {
  req0 <- 600 / 0.98 + 400 / 0.94; req5 <- 700 / 0.98 + 450 / 0.94
  hu0 <- 1100; add <- 60; avail <- 0.9
  gap <- req5 - (hu0 * avail + add * avail)
  expect_equal(gap, (req0 - hu0 * avail) + (req5 - req0) - add * avail)
})
cat("R unit tests passed\n")
