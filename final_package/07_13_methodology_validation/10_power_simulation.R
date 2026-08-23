#!/usr/bin/env Rscript
user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib != "" && !dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
if (user_lib != "") .libPaths(c(user_lib, .libPaths()))

# ============================================================================
# 10_power_simulation.R
# Statistical Power & Sample Sufficiency Simulation for Mediation Pathways
# Evaluates Monte Carlo power across N=207 highway sensors over 500 iterations.
# ============================================================================

metrics_file <- if (file.exists("metr_la_metrics.csv")) {
  "metr_la_metrics.csv"
} else if (file.exists("07_13_methodology_validation/metr_la_metrics.csv")) {
  "07_13_methodology_validation/metr_la_metrics.csv"
} else if (file.exists(file.path("..", "metr_la_metrics.csv"))) {
  file.path("..", "metr_la_metrics.csv")
} else {
  "metr_la_metrics.csv"
}

real_data <- read.csv(metrics_file)
real_data$reliability <- pmax(0, 1.0 - (0.6 * real_data$zero_rate + 0.2 * real_data$cusum_flag_rate + 0.2 * real_data$ewma_flag_rate))
real_data$disparity <- real_data$persistence_error

simulate_once <- function(n = 207, true_reliability_effect = 0.5) {
  idx <- sample(1:nrow(real_data), n, replace = TRUE)
  df <- real_data[idx, ]
  
  df$disparity <- true_reliability_effect * (1 - df$reliability) + 
                  0.01 * df$topology + 
                  0.01 * df$density + 
                  rnorm(n, sd = sd(real_data$disparity)*0.1)

  df$density_bin <- ifelse(df$density > median(df$density), "high_density", "low_density")
  return(df)
}


# ==========================================

N_SIM <- 500  # reduce to ~50 for a quick first test before committing to 500
TRUE_EFFECT <- 0.5

coverage <- logical(N_SIM)
ci_widths <- numeric(N_SIM)
point_estimates <- numeric(N_SIM)

for (i in seq_len(N_SIM)) {
  sim_data <- simulate_once(n = 207, true_reliability_effect = TRUE_EFFECT)
  
  # Standard mediation via lm
  sim_data$density_bin <- ifelse(sim_data$density_bin == "high_density", 1, 0)
  sim_data$traffic_regime <- ifelse(sim_data$traffic_regime == "congested", 1, 0)
  
  fit_direct <- lm(disparity ~ density_bin + reliability + topology + traffic_regime, data=sim_data)
  fit_rel <- lm(reliability ~ density_bin + traffic_regime, data=sim_data)
  
  a <- coef(fit_rel)["density_bin"]
  b <- coef(fit_direct)["reliability"]
  ie <- a * b
  
  se_a <- summary(fit_rel)$coefficients["density_bin", "Std. Error"]
  se_b <- summary(fit_direct)$coefficients["reliability", "Std. Error"]
  se_ie <- sqrt(b^2 * se_a^2 + a^2 * se_b^2)
  
  point_estimates[i] <- ie
  ci_lo <- ie - 1.96 * se_ie
  ci_hi <- ie + 1.96 * se_ie
  ci_widths[i] <- ci_hi - ci_lo
  coverage[i] <- (TRUE_EFFECT >= ci_lo) && (TRUE_EFFECT <= ci_hi)
}

# ==========================================

n_valid <- sum(!is.na(coverage))
cat("=== Power simulation results (n=207, N_SIM =", N_SIM, ") ===\n")
cat("Successful runs:", n_valid, "/", N_SIM, "\n")
if (n_valid > 0) {
  cat("Empirical CI coverage of true effect (target ~95%):",
      round(100 * mean(coverage, na.rm = TRUE), 1), "%\n")
  cat("Mean CI width:", round(mean(ci_widths, na.rm = TRUE), 3), "\n")
  cat("Mean point estimate (true value =", TRUE_EFFECT, "):",
      round(mean(point_estimates, na.rm = TRUE), 3), "\n")
  cat("SD of point estimates (estimator variability at n=207):",
      round(sd(point_estimates, na.rm = TRUE), 3), "\n")
}

cat("\n=== Interpretation ===\n")
cat("If coverage is well below 95%, the estimator's CIs are too narrow for\n")
cat("this sample size (overconfident). If mean CI width is large relative to\n")
cat("the effect size itself, n=207 may be too small for a precise estimate\n")
cat("even if coverage is technically correct. Either signals a need for a\n")
cat("simpler, more stable estimator (e.g. parametric mediation with linear/\n")
cat("logistic nuisance models) as a fallback for the real analysis.\n")

cat("\n=== NOTE ===\n")
cat("The field-name extraction logic in step 2 above (`ie_row <- s[...]`) is\n")
cat("a best guess and WILL likely need correcting once you see faircause's\n")
cat("actual summary() output structure from 02_ctf_estimation_faircause.R.\n")
cat("Run that script first, inspect `str(summary(result))`, then fix the\n")
cat("extraction logic here before trusting this simulation's numbers.\n")