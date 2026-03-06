* =============================================================================
* estimate_heterog_variance.do — Figures 3-4, A5 (Stata residuals step)
*
* Estimates teacher effects separately by student subgroup. For each binary
* covariate (white, male, disadv, atrisk), interacts the covariate with
* teacher FE (absorbing cov#teachid) so that each teacher gets a separate
* effect for each subgroup value (0 and 1).
*
* Residuals are collapsed to teacher-year level and reshaped wide with
* separate columns for each subgroup (e.g., testscores_r0, testscores_r1).
*
* Output: temp/teach_mean_resids_cov{white,male,disadv,atrisk}.dta
*   Used by vcov_hetero.py (Figures 3-4, A5) and vcov_hetero_subgroups.py
*   (Table A13).
*
* "atrisk" is constructed here by regressing criminal arrest on all covariates
* and splitting at the predicted-arrest median.
* =============================================================================
clear all
clear matrix
set more off

* Load data and options
do code/set_options.do
do code/preamble.do

* Loop over binary student covariates
foreach cov in white male disadv atrisk {
    global w "`cov'"

    if "${w}" == "atrisk" {
        qui reg aoc_crim ${covdesign} ${covsadj}
        predict arrest_hat
        su arrest_hat, d
        gen atrisk = arrest_hat >= r(p50)
    }

    drop if ${w} == .

    * Residuals
    foreach var of varlist aoc_any aoc_crim aoc_index aoc_incar aoc_traff testscores behavpca studypca gpa_weighted college_bound grad {
        reghdfe `var' ${covdesign}, abs(${w}#teachid) resid
		capture drop `var'_r
        predict `var'_r, dresiduals
    }

    * Collapse by teacher year
    preserve
    collapse (mean) *_r, by(teachid year ${w})
    drop if ${w} == .
    reshape wide *_r, i(teachid year) j(${w})

    sort teachid year
    by teachid: gen obs = _n
    drop year

    reshape wide *_r*, i(teachid) j(obs)

    save temp/teach_mean_resids_cov${w}.dta, replace
    restore
}
