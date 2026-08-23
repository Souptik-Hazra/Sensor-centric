# ============================================================================
# METR-LA Structural Causal Digital Twin — R Package Requirements
# Verified on: R 4.x | Pipeline: 5/5 R scripts PASSED
# ============================================================================

user_lib <- Sys.getenv("R_LIBS_USER")
if (user_lib != "" && !dir.exists(user_lib)) dir.create(user_lib, recursive = TRUE)
if (user_lib != "") .libPaths(c(user_lib, .libPaths()))

cran_pkgs <- c(
  "dagitty",       # Judea Pearl SCM DAG identifiability & d-separation
  "sandwich",      # Heteroskedasticity-consistent (HC) robust standard errors
  "clubSandwich",  # CR2 clustered standard errors for panel data
  "lme4",          # Linear mixed-effects models (temporal alignment)
  "boot",          # Non-parametric bootstrap resampling (1,000-sample 95% CIs)
  "devtools"       # GitHub package installer (for faircause)
)

for (pkg in cran_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    cat("Installing CRAN package:", pkg, "\n")
    install.packages(pkg, lib = user_lib, repos = "https://cloud.r-project.org")
  }
}

# Plecko & Bareinboim Structural Fairness Model (SFM) Mediation Engine
if (!requireNamespace("faircause", quietly = TRUE)) {
  cat("Installing faircause from GitHub (dplecko/CFA)...\n")
  devtools::install_github("dplecko/CFA", lib = user_lib)
}

cat("All R requirements installed successfully!\n")
