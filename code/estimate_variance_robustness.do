* =============================================================================
* estimate_variance_robustness.do — Figure A4 (per-specification residuals)
*
* Called by run_robustness.sh once per covariate specification. Each call
* receives a different covariate set via command-line arguments:
*   arg 1 (local `1'): iteration number
*   arg 2+ (local `0'): the covariate specification string (from robust_options.txt)
*
* The script:
*   1. Overrides $covdesign with the specification passed via command line
*   2. Runs areg for test scores, behaviors, and criminal arrest
*   3. Collapses to teacher-year means and reshapes wide
*   4. Saves residuals to temp/robust/resids_iter{N}.dta
*
* These files are later read by vcov_robustness.py, which computes 1-SD
* effects across all 1,369 specifications and produces Figure A4.
* =============================================================================
clear all
clear matrix
set more off
capture ssc install tuples
set seed 617238

* Parse command-line arguments: iteration number and covariate specification.
* `0' contains the full argument string (e.g., "42 i.year##i.grade ...").
* The covdesign global is set from `0' after stripping the leading iteration number.
global outcome = "testscores"
global covdesign = "`0'"
global covdesign = regexr("$covdesign", "^[0-9]+","")
di "Working with covdesign ${covdesign}"
dis "-------------"

global covsadj = "pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal"
local iter = `1'
di "Working on iteration `iter'"

* Load data and options (note: set_options.do will re-set $covdesign, but we
* have already overridden it above; preamble.do uses the overridden value)
do code/set_options.do
do code/preamble.do

* Get the residuals
foreach var of varlist aoc_crim testscores behavpca {
    areg `var' $covdesign, abs(teachid)   
    predict `var'_r, dresiduals 
}

* Collapse by teacher year
preserve
collapse (mean) *_r, by(teachid year)

sort teachid year
by teachid: gen obs = _n

drop year
reshape wide *_r, i(teachid) j(obs)

* save to Scratch
save temp/robust/resids_iter`iter'.dta, replace


