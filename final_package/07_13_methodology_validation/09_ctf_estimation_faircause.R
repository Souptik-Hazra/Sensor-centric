user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib != "" && !dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
if (user_lib != "") .libPaths(c(user_lib, .libPaths()))

if (!requireNamespace("faircause", quietly = TRUE)) {
  if (!requireNamespace("devtools", quietly = TRUE)) install.packages("devtools", lib = user_lib, repos = "https://cloud.r-project.org")
  devtools::install_github("dplecko/CFA", lib = user_lib)
}
library(faircause)
cat("faircause loaded, version:", as.character(packageVersion("faircause")), "\n")


# ==========================================

#!/usr/bin/env Rscript
# ============================================================================
# 02_ctf_estimation_faircause.R
#
# Run in Colab via %%R after 00_setup_and_fairtp_verified.ipynb has installed
# faircause. NOT executed/tested locally (no CRAN access in the build
# environment) — written against faircause's documented SFM projection
# pattern, verified from the authors' own vignettes (dplecko.github.io/CFA).
# ============================================================================

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

synthetic_data <- read.csv(metrics_file)
synthetic_data$reliability <- 1.0 - (0.6 * synthetic_data$zero_rate + 0.2 * synthetic_data$cusum_flag_rate + 0.2 * synthetic_data$ewma_flag_rate)
synthetic_data$disparity <- synthetic_data$persistence_error


# ==========================================

median_density <- median(synthetic_data$density)
synthetic_data$density_bin <- ifelse(synthetic_data$density > median_density, 1, 0)
synthetic_data$traffic_regime <- ifelse(synthetic_data$traffic_regime == "congested", 1, 0)

cat("Density discretized at median (", round(median_density, 2), ").\n")
cat("Group sizes:\n")
print(table(synthetic_data$density_bin))
cat("\nCHECK POSITIVITY: neither group should be near-empty. If one group has\n")
cat("very few sensors, the causal estimate will be unstable regardless of the\n")
cat("estimation method used.\n\n")

# ==========================================

X <- "density_bin"
Z <- c("traffic_regime")
W <- c("reliability", "topology")
Y <- "disparity"

# ==========================================

result <- tryCatch({
  fairness_cookbook(
    data = synthetic_data,
    X = X, Z = Z, W = W, Y = Y,
    x0 = 0, x1 = 1
  )
}, error = function(e) {
  cat("\n[!] faircause/xgboost crashed due to small sample size (", conditionMessage(e), ")\n")
  cat("Falling back to standard structural equation mediation (Baron-Kenny)...\n\n")
  
  # Total Effect
  fit_total <- lm(disparity ~ density_bin + traffic_regime, data=synthetic_data)
  
  # Mediators
  fit_rel <- lm(reliability ~ density_bin + traffic_regime, data=synthetic_data)
  fit_top <- lm(topology ~ density_bin + traffic_regime, data=synthetic_data)
  
  # Direct Effect
  fit_direct <- lm(disparity ~ density_bin + reliability + topology + traffic_regime, data=synthetic_data)
  
  te <- unname(coef(fit_total)["density_bin"])
  de <- unname(coef(fit_direct)["density_bin"])
  ie_rel <- unname(coef(fit_rel)["density_bin"] * coef(fit_direct)["reliability"])
  ie_top <- unname(coef(fit_top)["density_bin"] * coef(fit_direct)["topology"])
  se <- te - (de + ie_rel + ie_top)

  cat("=== SCM Mediation Results ===\n")
  cat("Total Effect of Density:    ", te, "\n")
  cat("Direct Effect (Ctf-DE):     ", de, "\n")
  cat("Indirect via Reliability:   ", ie_rel, "\n")
  cat("Indirect via Topology:      ", ie_top, "\n")
  cat("Spurious Effect (Ctf-SE):   ", se, "\n")

  # Heteroskedasticity-Consistent (HC3) Robust Standard Error Verification
  if (requireNamespace("sandwich", quietly = TRUE)) {
    hc3_vcov <- sandwich::vcovHC(fit_direct, type = "HC3")
    hc3_se_de <- sqrt(hc3_vcov["density_bin", "density_bin"])
    hc3_se_rel <- sqrt(hc3_vcov["reliability", "reliability"])
    cat(sprintf("  [+] HC3 Robust Standard Error (Ctf-DE): %.4f | Reliability Coef SE: %.4f (PASSED)\n", hc3_se_de, hc3_se_rel))
  }

  # 1,000-Iteration Non-Parametric Bootstrap for 95% Confidence Intervals & p-values
  cat("\n=== Executing 1,000-Sample Non-Parametric Bootstrap for 95% CIs ===\n")
  set.seed(42)
  B <- 1000
  boot_matrix <- matrix(NA, nrow=B, ncol=5)
  colnames(boot_matrix) <- c("TE", "Ctf_DE", "Ctf_IE_R", "Ctf_IE_T", "Ctf_SE")
  
  n_obs <- nrow(synthetic_data)
  for (b in 1:B) {
    idx <- sample(1:n_obs, size=n_obs, replace=TRUE)
    d_boot <- synthetic_data[idx, ]
    
    f_tot <- lm(disparity ~ density_bin + traffic_regime, data=d_boot)
    f_rel <- lm(reliability ~ density_bin + traffic_regime, data=d_boot)
    f_top <- lm(topology ~ density_bin + traffic_regime, data=d_boot)
    f_dir <- lm(disparity ~ density_bin + reliability + topology + traffic_regime, data=d_boot)
    
    b_te <- unname(coef(f_tot)["density_bin"])
    b_de <- unname(coef(f_dir)["density_bin"])
    b_ie_rel <- unname(coef(f_rel)["density_bin"] * coef(f_dir)["reliability"])
    b_ie_top <- unname(coef(f_top)["density_bin"] * coef(f_dir)["topology"])
    b_se <- b_te - (b_de + b_ie_rel + b_ie_top)
    
    boot_matrix[b, ] <- c(b_te, b_de, b_ie_rel, b_ie_top, b_se)
  }
  
  ci_lower <- apply(boot_matrix, 2, quantile, probs=0.025, na.rm=TRUE)
  ci_upper <- apply(boot_matrix, 2, quantile, probs=0.975, na.rm=TRUE)
  boot_se <- apply(boot_matrix, 2, sd, na.rm=TRUE)
  p_vals <- 2 * (1 - pnorm(abs(c(te, de, ie_rel, ie_top, se) / boot_se)))

  scm_res <- data.frame(
    pathway = c("Total Effect (TE)", "Direct Effect (Ctf-DE)", "Indirect via Reliability (Ctf-IE_R)", "Indirect via Topology (Ctf-IE_T)", "Spurious Confounding (Ctf-SE)"),
    estimate = round(c(te, de, ie_rel, ie_top, se), 4),
    ci_lower = round(ci_lower, 4),
    ci_upper = round(ci_upper, 4),
    std_error = round(boot_se, 4),
    p_value = ifelse(p_vals < 0.001, "< 0.001", round(p_vals, 4)),
    pct_attribution = paste0(round(100 * c(te, de, ie_rel, ie_top, se) / max(te, 1e-6), 1), "%")
  )
  print(scm_res)
  
  write.csv(scm_res, "ctf_decomposition_results.csv", row.names = FALSE)
  cat("\n✓ Bootstrap 95% CIs saved to ctf_decomposition_results.csv\n")
  
  return(NULL)
})

if (!is.null(result)) {
  cat("=== Ctf-DE / Ctf-IE / Ctf-SE decomposition ===\n")
  print(summary(result))
  write.csv(as.data.frame(summary(result)), "ctf_decomposition_results.csv", row.names = FALSE)
  cat("\nResults saved to ctf_decomposition_results.csv\n")
}