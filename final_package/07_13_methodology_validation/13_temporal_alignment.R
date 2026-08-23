#!/usr/bin/env Rscript
# ============================================================================
# 06_temporal_alignment.R
#
# Checks whether combining static (density, topology) and time-varying
# (reliability) mediators in one cross-sectional faircause estimation is
# valid, or whether the panel structure (repeated sensor-windows from the
# same physical sensor) needs explicit handling.
#
# NOT executed/tested locally. Written against documented package APIs
# (sandwich, clubSandwich) — verify on first Colab run.
# ============================================================================

user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib != "" && !dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
if (user_lib != "") .libPaths(c(user_lib, .libPaths()))

ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}

ensure_pkg("sandwich")
ensure_pkg("clubSandwich")
ensure_pkg("lme4")

library(sandwich)
library(clubSandwich)
library(lme4)

# ==========================================

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
n_sensors <- nrow(real_data)
n_windows <- 12

set.seed(42)
base_reliability <- pmax(0, 1.0 - (0.6 * real_data$zero_rate + 0.2 * real_data$cusum_flag_rate + 0.2 * real_data$ewma_flag_rate))
base_disparity <- real_data$persistence_error

grid <- expand.grid(window = 1:n_windows, sensor_idx = 1:n_sensors)
n_rows <- nrow(grid)

rel_noise <- rnorm(n_rows, mean = 0, sd = 0.05)
disp_noise <- rnorm(n_rows, mean = 0, sd = base_disparity[grid$sensor_idx] * 0.1)

panel_data <- data.frame(
  sensor_id   = real_data$node_id[grid$sensor_idx],
  window      = grid$window,
  density     = real_data$density[grid$sensor_idx],
  topology    = real_data$topology[grid$sensor_idx],
  reliability = pmin(pmax(base_reliability[grid$sensor_idx] + rel_noise, 0), 1),
  disparity   = base_disparity[grid$sensor_idx] + disp_noise
)
cat("Panel data simulated from REAL METR-LA base:", nrow(panel_data), "rows,", n_sensors, "sensors x", n_windows, "windows\n")


# ==========================================

naive_model <- lm(disparity ~ reliability + topology + density, data = panel_data)
cat("\n=== Naive OLS (ignoring panel structure) ===\n")
print(summary(naive_model)$coefficients)

# ==========================================

clustered_se <- coef_test(naive_model, vcov = "CR2", cluster = panel_data$sensor_id)
cat("\n=== Same model, sensor-clustered standard errors ===\n")
print(clustered_se)

cat("\nCompare the naive SEs above to these clustered SEs. If clustered SEs\n")
cat("are meaningfully larger, the naive cross-sectional approach was\n")
cat("overconfident — its p-values and CIs cannot be trusted as reported.\n")

# ==========================================

mixed_model <- lmer(disparity ~ reliability + topology + density + (1 | sensor_id),
                     data = panel_data)
cat("\n=== Mixed-effects model (random intercept per sensor) ===\n")
print(summary(mixed_model))

# ==========================================

cat("\n=== Implications for faircause ===\n")
cat("faircause's fairness_cookbook() does not natively support clustered SEs\n")
cat("or mixed-effects structures — it assumes i.i.d. rows. Three options:\n\n")
cat("Option A (simplest, recommended first pass): collapse each sensor's\n")
cat("  multiple windows into ONE row (e.g. mean reliability across windows)\n")
cat("  before running faircause. Loses the time-varying signal but keeps\n")
cat("  estimation valid and simple.\n\n")
cat("Option B (if time-variation matters): run faircause SEPARATELY per\n")
cat("  window, then pool the resulting Ctf-DE/IE/SE estimates across windows\n")
cat("  using a random-effects meta-analysis (e.g. `metafor` package) rather\n")
cat("  than pooling the raw data.\n\n")
cat("Option C (most rigorous, most work): a full g-computation/marginal\n")
cat("  structural model approach via `gfoRmula` or `ipw`, treating this as\n")
cat("  a genuine longitudinal causal inference problem. Only worth it if\n")
cat("  reliability's time-variation is itself a key part of the story you\n")
cat("  want to tell, not just a nuisance to control for.\n")

# ==========================================

cat("\n=== Effect on the spurious effect (Ctf-SE) component specifically ===\n")
cat("Unmodeled within-sensor correlation acts like an unobserved confounder\n")
cat("shared across a sensor's windows (sensor_effect above). If this\n")
cat("correlates with density (e.g. older/cheaper sensors cluster in\n")
cat("low-density areas AND have a persistent unobserved quality issue),\n")
cat("failing to account for it will typically INFLATE the apparent Ctf-SE\n")
cat("(spurious effect), because some of what looks like 'confounding via C'\n")
cat("is actually 'confounding via unmodeled sensor identity.' This is a\n")
cat("plausible-direction argument, not a proven universal result — the\n")
cat("actual bias direction depends on how sensor_effect correlates with\n")
cat("density in the real data, which should be checked empirically:\n")
cat("  cor(tapply(panel_data$sensor_effect, panel_data$sensor_id, mean),\n")
cat("      tapply(panel_data$density, panel_data$sensor_id, mean))\n")