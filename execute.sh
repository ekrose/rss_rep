#!/bin/bash
# =============================================================================
# Master execution script for Rose, Schellenberg, and Shem-Tov (Econometrica 2026)
#
# Usage:
#   bash execute.sh
#
# See README for complete instructions. Important steps are:
#   - Python virtual environment set up: python -m venv .venv && pip install -r requirements.txt
#   - Stata SE installed (path configured via STATA in .envrc)
#   - Environment variable PROJECT_DATA_DIR set (sourced from .envrc)
#   - Directory $PROJECT_DATA_DIR exists
#   - Note, if you have anaconda installed you might nedd to do "conda deactivate" before
#     running the script but it can work also without depneding on how anaconda is setup.
#
# This script runs all analysis code in the correct order. Each block is
# annotated with the table(s) or figure(s) it produces. Many outputs require
# a Stata step (generating teacher-level residuals in temp/) followed by a
# Python step (computing U-statistic estimates and producing final output).
#
# Output:
#   - LaTeX tables  -> tables/*.tex
#   - Figures        -> figures/*.pdf, figures/*.png
#   - In-text stats  -> tables/in_text_citations.txt
# =============================================================================

source .envrc
source .venv/bin/activate

# Run Stata in batch mode (-b) and wait for it to finish.
# On macOS, -b may return immediately (Stata forks), so we poll the log.
# We wait until the log file stops growing AND contains "end of do-file",
# which avoids premature exit from sub-file markers (set_options.do, etc.)
# that appear while the main do-file is still running.
run_stata() {
    local dofile="$1"
    shift
    local logbase
    logbase=$(basename "$dofile" .do)
    rm -f "${logbase}.log" 2>/dev/null
    "$STATA" -b do "$dofile" "$@"
    # Wait for log file to appear
    while [ ! -f "${logbase}.log" ]; do
        sleep 2
    done
    # Wait until the log file stops growing and contains "end of do-file"
    local prev_size=0
    while true; do
        sleep 3
        local curr_size
        curr_size=$(wc -c < "${logbase}.log" 2>/dev/null || echo 0)
        if [ "$curr_size" = "$prev_size" ] && grep -q "end of do-file" "${logbase}.log" 2>/dev/null; then
            break
        fi
        prev_size=$curr_size
    done
}

# Clean up stale logs and intermediate files from previous runs
rm -f *.log 2>/dev/null
rm -f temp/*.dta temp/*.csv temp/*.smcl temp/robust/*.dta temp/robust/*.log 2>/dev/null

# =============================================================================
# DATA SIMULATION
# =============================================================================
# This step was done by the authors and does not need to be run now. 
# We included the code and the commands we executed (commented out) below for reference
# When using replication archive, you can skip this section and place the analysis file at
# $PROJECT_DATA_DIR/analysis.dta and the teacher characteristics file at
# data/teacher_covars.dta.
# =============================================================================

# -----------------------------------------------------------------------------
# Step 0a: Generate simulated analysis data
# Reads summary statistics from data/summary.pkl and writes analysis.dta
# Note, due to privacy restriction, the summary.pkl file is not part of the replication package. 
#		however, the code to create it from the real analysis data is included in the replication archive and it is "fake_data_formats.py"
# to $PROJECT_DATA_DIR.
# -----------------------------------------------------------------------------
# python code/simulate_from_summary.py

# -----------------------------------------------------------------------------
# Step 0b: Generate simulated teacher characteristics
# Reads teacher-year pairs from $PROJECT_DATA_DIR/analysis.dta and writes
# data/teacher_covars.dta with simulated demographics, education, test
# scores, and pay. Used by teacher_chars.do (Tables A1, A12).
# -----------------------------------------------------------------------------
# python code/simulate_teacher_covars.py

# =============================================================================
# ANALYSIS
# =============================================================================

# -----------------------------------------------------------------------------
# Step 1: Summary statistics
# Tables 1 and A11
# -----------------------------------------------------------------------------
run_stata code/summary_stats.do

# -----------------------------------------------------------------------------
# Step 2: Main variance/covariance estimates
# Tables 2 and 3, Figure A1
#   - Stata: regresses outcomes on covariates absorbing teacher FE, saves
#     teacher-year residuals to temp/teach_mean_resids.dta (wide format)
#   - Python part 1: computes U-statistic variance/covariance estimates
#   - Python part 2: computes standard errors and formats tables
# -----------------------------------------------------------------------------
run_stata code/estimate_variance.do
python code/vcov_main_part1.py
python code/vcov_main_part2.py

# -----------------------------------------------------------------------------
# Step 3: Implied regression decomposition
# Tables 4 and A8
#   - Uses residuals from Step 2 (temp/teach_mean_resids.dta and
#     temp/teachSchl_mean_resids.dta)
#   - Estimates multivariate infeasible regression of long-run outcomes
#     on short-run teacher effects via U-statistics
# -----------------------------------------------------------------------------
python code/vcov_implied_reg.py

# -----------------------------------------------------------------------------
# Step 4: Entry school instrumental variables
# Table 5
# -----------------------------------------------------------------------------
run_stata code/entry_ivs_part1.do

# -----------------------------------------------------------------------------
# Step 5: Variance by time gap
# Figure 1
#   - Uses residuals from Step 2
# -----------------------------------------------------------------------------
python code/vcov_main_part3.py

# -----------------------------------------------------------------------------
# Step 6: Omitted variable bias analysis
# Figure 2, Figure A3, Tables A4 and A5
# -----------------------------------------------------------------------------
run_stata code/estimate_ovb.do

# -----------------------------------------------------------------------------
# Step 7: Heterogeneous variance estimates
# Figures 3-4 and A5
#   - Stata: estimates residuals separately by subgroup
#   - Python: computes subgroup-specific variance/covariance via U-statistics
# -----------------------------------------------------------------------------
run_stata code/estimate_heterog_variance.do
python code/vcov_hetero.py

# -----------------------------------------------------------------------------
# Step 8: Bottom X percent analysis
# Figure 5
#   - Uses residuals from Step 2
# -----------------------------------------------------------------------------
python code/bottomXpercent.py

# -----------------------------------------------------------------------------
# Step 9: Variance by year gap (additional)
# Figure A2
#   - Uses residuals from Step 2
# -----------------------------------------------------------------------------
python code/vcov_main_part4.py

# -----------------------------------------------------------------------------
# Step 10: Robustness to covariate specification
# Figure A4
#   - Run separately: bash run_robustness.sh [n_workers]
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Step 11: Teacher characteristics
# Tables A1 and A12
#   - Uses data/teacher_covars.dta (generated in Step 0b)
# -----------------------------------------------------------------------------
run_stata code/teacher_chars.do

# -----------------------------------------------------------------------------
# Step 12: Regression-based variance decomposition
# Tables A2 and A3
# -----------------------------------------------------------------------------
run_stata code/regression_version.do

# -----------------------------------------------------------------------------
# Step 13: Additional IV specifications
# Tables A6 and A7
# -----------------------------------------------------------------------------
run_stata code/entry_ivs_part2.do
run_stata code/entry_ivs_part3.do

# -----------------------------------------------------------------------------
# Step 14: Within-school variance decomposition
# Table A9
#   - Uses residuals from Step 2 (including school FE identifiers)
#   - Computes within-school U-statistic estimates by iterating over schools
# -----------------------------------------------------------------------------
python code/vcov_within_schl.py

# -----------------------------------------------------------------------------
# Step 15: Sensitivity to time horizon
# Table A10
#   - Uses residuals from Step 2
# -----------------------------------------------------------------------------
python code/vcov_t_sensitivity.py

# -----------------------------------------------------------------------------
# Step 16: Heterogeneous variance by demographic subgroups
# Table A13
#   - Python: estimates subgroup-specific variance/covariance
#   - Stata: formats the final table
# -----------------------------------------------------------------------------
python code/vcov_hetero_subgroups.py
run_stata code/vcov_hetero_subgroups_table.do

# -----------------------------------------------------------------------------
# Step 17: Covariate correlations (in-text calculations)
#   - Stata: computes correlations between teacher effects and covariates
#   - Python: aggregates and formats for in-text citations
# -----------------------------------------------------------------------------
run_stata code/covariate_correlation_part1.do
python code/covariate_correlation_part2.py
