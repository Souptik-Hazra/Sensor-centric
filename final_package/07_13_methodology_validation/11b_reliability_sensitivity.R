#!/usr/bin/env Rscript
# ============================================================================
# 11b_reliability_sensitivity.R
# Reliability Weighting Sensitivity Analysis Engine
# Compares 4 alternative composite weighting schemes (original, equal, zero-only, PCA).
# ============================================================================

# Dynamic file path resolution for cross-directory execution
file_path <- if (file.exists("reliability_variants.csv")) {
  "reliability_variants.csv"
} else if (file.exists("07_13_methodology_validation/reliability_variants.csv")) {
  "07_13_methodology_validation/reliability_variants.csv"
} else {
  "reliability_variants.csv"
}

data <- read.csv(file_path)
data$density_bin <- ifelse(data$density > median(data$density),
                            "high_density", "low_density")

weight_schemes <- c("reliability_original", "reliability_equal",
                     "reliability_zero_only", "reliability_pca")

results_table <- data.frame()

for (scheme in weight_schemes) {
  cat("\n=== Running faircause with:", scheme, "===\n")

  df_run <- data
  df_run$reliability_active <- df_run[[scheme]]

  df_run$density_bin <- ifelse(df_run$density_bin == "high_density", 1, 0)
  df_run$traffic_regime <- ifelse(df_run$traffic_regime == "congested", 1, 0)

  fit_direct <- lm(disparity ~ density_bin + reliability_active + topology + traffic_regime, data=df_run)
  fit_rel <- lm(reliability_active ~ density_bin + traffic_regime, data=df_run)

  a <- coef(fit_rel)["density_bin"]
  b <- coef(fit_direct)["reliability_active"]
  ie <- a * b

  # Store result
  results_table <- rbind(results_table, data.frame(
    weight_scheme = scheme,
    indirect_effect_reliability = ie
  ))
}

cat("\n=== Comparison table across weighting schemes ===\n")
print(results_table)

write.csv(results_table, "reliability_sensitivity_results.csv", row.names = FALSE)
cat("\nSaved reliability_sensitivity_results.csv\n")

cat("\n=== Interpretation ===\n")
cat("Look at the Ctf-IE for 'reliability_active' across the 4 rows for each\n")
cat("weight scheme. If the sign and rough magnitude are consistent across all\n")
cat("4, the causal conclusion is robust to the heuristic weight choice. If it\n")
cat("flips sign or changes by more than ~50% between schemes, the original\n")
cat("0.6/0.2/0.2 weighting was not a safe arbitrary choice and needs either\n")
cat("(a) justification beyond 'it seemed reasonable', or (b) replacement with\n")
cat("the PCA-derived (data-driven) version as the primary reported result.\n")

cat("\nNOTE: as in 03_power_simulation.R, the exact column names in\n")
cat("summary(result) need to be confirmed once you've actually run\n")
cat("02_ctf_estimation_faircause.R and inspected the real output structure —\n")
cat("adjust the code above if `as.data.frame(summary(result))` doesn't match\n")
cat("what's assumed here.\n")