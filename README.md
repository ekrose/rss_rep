# Replication Archive

**"The Effects of Teacher Quality on Adult Criminal Justice Contact"**
Rose, Schellenberg, and Shem-Tov, *Econometrica*, 2026

## Overview

This repository contains the replication code for all tables and figures in the paper. Due to data privacy restrictions, the original administrative data cannot be shared; simulated data is provided instead (see [Data Availability Statement](#data-availability-statement) and [Simulated data](#simulated-data)).

**Results produced from the simulated data will not reproduce the estimates reported in the paper.**

## Data Availability Statement

The analysis uses confidential administrative data from two sources that are **not** publicly available and cannot be shared in this replication archive due to data privacy restrictions. The archive instead provides simulated data that preserves the structure and approximate distributions of the original data (see [Simulated data](#simulated-data) below). Results produced from the simulated data will not reproduce the estimates reported in the paper.

### Datasets included in this archive

| File | Description |
|---|---|
| `data/analysis.dta` (also available as csv) | Simulated student-level analysis dataset (10,000 observations) |
| `data/teacher_covars.dta` (also available as csv) | Simulated teacher characteristics dataset |

### Datasets excluded from this archive

**1. North Carolina Education Research Data Center (NCERDC)**
North Carolina public school administrative records, including student test scores, demographic characteristics, and teacher employment records. The NCERDC data do not carry a formal version number; the relevant cohorts and years are described in the paper. These data are confidential and governed by a data use agreement with NCERDC.

**2. North Carolina Administrative Office of the Courts (NCAOC)**
Administrative records of adult criminal justice contact for North Carolina residents, linked to the NCERDC records. These data were acquired directly from the AOC.

### Accessing the original data

**NCERDC**: Independent researchers may apply for access to North Carolina public school administrative records through the North Carolina Education Research Data Center, housed at Duke University's Center for Child and Family Policy. Access requires an approved research proposal and executed data use agreement. Applications are subject to institutional review. See [https://childandfamilypolicy.duke.edu/project/nc-education-research-data-center/](https://childandfamilypolicy.duke.edu/project/nc-education-research-data-center/) for current application procedures, requirements, and fees. The monetary cost and processing time for new applications are set by NCERDC.

**NCAOC**: Researchers seeking access to North Carolina criminal justice administrative records should contact the North Carolina Administrative Office of the Courts directly. These data are not routinely available to outside researchers and may require a data use agreement.

### Data citations

North Carolina Education Research Data Center (NCERDC). *North Carolina Public School Administrative Records*. Durham, NC: Duke University Center for Child and Family Policy. 

North Carolina Administrative Office of the Courts (NCAOC). *Administrative Records of Criminal Justice Contact*. Raleigh, NC: North Carolina Administrative Office of the Courts.

---

## Requirements

### Software

- **Stata SE** version 19 or later (required Stata packages `gtools`, `ivreg2`, and `tuples` are installed automatically)
- **Python** 3.14+ (see `.python-version`)

### Python environment setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: if you have Anaconda installed, you may need to run `conda deactivate` before activating the virtual environment, depending on your Anaconda configuration.

### Environment variable

The analysis data file (`analysis.dta`) is read from the path specified by the environment variable `PROJECT_DATA_DIR`. This is set in `.envrc` (compatible with [direnv](https://direnv.net/)):

```
export PROJECT_DATA_DIR="$HOME/Documents/GitHub/rsss_rep/data/"
```

If you do not use direnv, export this variable manually before running any code. The directory must exist.

## Repository structure

```
├── execute.sh                  # Main script: runs all code in order
├── run_robustness.sh           # Runs robustness specifications (Figure A4)
├── requirements.txt            # Python package dependencies
├── LICENSE                     # BSD 3-Clause license
├── .envrc                      # Environment variable (PROJECT_DATA_DIR)
├── .python-version             # Python version specification
│
├── code/
│   ├── set_options.do          # Stata globals: covdesign, data path
│   ├── preamble.do             # Stata preamble: derived variables, sample restrictions, VAM
│   ├── vam.ado                 # Stata ado: value-added model program
│   ├── funcs_vcov_ustats.py    # Python library: U-statistic variance/covariance estimators
│   ├── robust_options.txt      # 812 covariate specifications for robustness checks
│   │
│   ├── simulate_from_summary.py    # (Reference) Generates simulated data from summary statistics
│   ├── simulate_teacher_covars.py  # (Reference) Generates simulated teacher characteristics
│   ├── fake_data_formats.py        # (Reference) Created summary.pkl from real data
│   │
│   ├── summary_stats.do            # Tables 1, A11
│   ├── estimate_variance.do        # Tables 2-3, Figure A1 (Stata residuals)
│   ├── vcov_main_part1.py          # Tables 2-3, Figure A1 (U-stat estimation)
│   ├── vcov_main_part2.py          # Tables 2-3 (standard errors)
│   ├── vcov_implied_reg.py         # Tables 4, A8
│   ├── entry_ivs_part1.do          # Table 5
│   ├── vcov_main_part3.py          # Figure 1
│   ├── estimate_ovb.do             # Figure 2, A3; Tables A4-A5
│   ├── estimate_heterog_variance.do    # Figures 3-4, A5 (Stata residuals)
│   ├── vcov_hetero.py                  # Figures 3-4, A5 (U-stat estimation)
│   ├── bottomXpercent.py           # Figure 5
│   ├── vcov_main_part4.py          # Figure A2
│   ├── estimate_variance_robustness.do # Figure A4 (Stata residuals, per specification)
│   ├── vcov_robustness.py              # Figure A4 (aggregation and plotting)
│   ├── teacher_chars.do            # Tables A1, A12
│   ├── regression_version.do       # Tables A2-A3
│   ├── entry_ivs_part2.do          # Table A6
│   ├── entry_ivs_part3.do          # Table A7
│   ├── vcov_within_schl.py         # Table A9
│   ├── vcov_t_sensitivity.py       # Table A10
│   ├── vcov_hetero_subgroups.py        # Table A13 (estimation)
│   ├── vcov_hetero_subgroups_table.do  # Table A13 (table formatting)
│   ├── covariate_correlation_part1.do  # In-text calculations (Stata)
│   └── covariate_correlation_part2.py  # In-text calculations (Python)
│
├── build code/
│   ├── build_final.py          # (Reference) Builds analysis.dta from raw NCERDC files
│   └── build_teacher_chars.py  # (Reference) Builds teacher_covars.dta from raw NCERDC files
│
├── data/
│   ├── analysis.dta            # Simulated analysis dataset (10,000 observations)
│   └── teacher_covars.dta      # Simulated teacher characteristics
│
├── tables/                     # Output: LaTeX tables (.tex)
├── figures/                    # Output: Figures (.pdf, .png)
└── temp/                       # Intermediate Stata datasets used between steps
```

## How to run

### Full replication

Run `execute.sh` from the repository root. This script runs all analysis code in the correct order:

```bash
source .venv/bin/activate
bash execute.sh
```

To capture all output to a log file:

```bash
PYTHONUNBUFFERED=1 bash execute.sh > log_master.txt 2>&1
```

Each block in `execute.sh` is annotated with the table or figure it produces. Many outputs require both a Stata step (which generates teacher-level residuals saved to `temp/`) and a Python step (which computes U-statistic estimates and produces the final table or figure).

### Robustness specifications (Figure A4)

The robustness analysis (`estimate_variance_robustness.do`) is designed to be run once per covariate specification. The file `code/robust_options.txt` contains 812 specifications. Use `run_robustness.sh` to loop over them:

```bash
# Run all 812 specifications with 4 parallel workers (default)
bash run_robustness.sh

# Run with 8 parallel workers
bash run_robustness.sh 8

# Run only the first 10 specifications (for testing)
bash run_robustness.sh 4 10
```

Each iteration runs Stata and saves residuals to `temp/robust/`. After all iterations complete, run `python code/vcov_robustness.py` to aggregate results and produce Figure A4.

## Output

- **Tables**: LaTeX files written to `tables/` (e.g., `table1.tex`, `tableA8.tex`)
- **Figures**: PDF and PNG files written to `figures/` (e.g., `figure1a.pdf`, `figure4a.png`)
- **In-text citations**: Written to `tables/in_text_citations.txt`

## Code architecture

The analysis pipeline has a recurring two-step structure:

1. **Stata step**: Loads student-level data, runs regressions absorbing teacher fixed effects, extracts residuals, collapses to teacher-year level, and saves to `temp/`.
2. **Python step**: Reads teacher-year residuals from `temp/`, computes variance and covariance of teacher effects using U-statistic estimators (implemented in `funcs_vcov_ustats.py`), estimates standard errors via analytical formulas or parametric bootstrap, and writes output tables/figures.

Key shared components:
- `code/set_options.do` — Defines the baseline covariate specification (`covdesign`) and loads the analysis dataset. Sourced by all `.do` files.
- `code/preamble.do` — Constructs derived variables (PCA indices, test score composites, behavioral composites), applies sample restrictions (drops teachers with fewer than 2 years per subject), and computes value-added models. Sourced by most `.do` files.
- `code/funcs_vcov_ustats.py` — Implements all U-statistic estimators: `varcovar()` for variance/covariance, `sd_effect_func()` for 1-SD effects, and associated standard error functions. Imported by all analysis `.py` files.

## Simulated data

The simulated dataset was generated by `code/simulate_from_summary.py`, which reads summary statistics from `data/summary.pkl` and produces a dataset with the same variable names and approximate marginal distributions as the real data. The simulation enforces some constraints to allow the analysis code to run on a sample of 10,000 observations instead of the millions of observations that are in the analysis data. These are simplifications for simulating the data but are not constraints imposed in the analysis or assumptions on the DGP. They just make the code able to work on a much smaller sample size and have enough variation in the small data to not generate any errors:

- Each teacher appears exactly 4 times (4 student observations per teacher)
- Each teacher is observed in exactly 2 distinct years with the same subject
- Half of teachers appear in 2 schools; the other half in 1 school
- Student IDs (`mastid`) are unique per row
- Twin IDs (`twin_id`) are shared by exactly 2 rows
- Binary indicators are simulated as Bernoulli draws

The simulated data is included in the archive as `data/analysis.dta`. Teacher characteristics are included as `data/teacher_covars.dta`. Both datasets are also available in `data/` as csvs.

Note: `data/summary.pkl` and the raw NCERDC data used to generate the simulated data are not included in the replication archive due to privacy restrictions. The code used to create them (`code/fake_data_formats.py`, `code/simulate_from_summary.py`, `code/simulate_teacher_covars.py`, and the scripts in `build code/`) is included for reference.



