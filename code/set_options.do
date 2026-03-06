* =============================================================================
* set_options.do
* Sourced by all .do files at the start of execution.
* Defines the two global covariate lists used throughout the analysis:
*   $covdesign — "design" controls included in all regressions (year-grade-subject
*                interactions, lagged scores, school/grade-year-teacher-subject means,
*                exceptionality indicators, demographics, and behavioral lags)
*   $covsadj  — "excluded" adjustment controls used in sensitivity/OVB analyses
*                (parental education dummies and twice-lagged test scores)
* Also loads the main analysis dataset and installs required Stata packages.
* =============================================================================

* Design controls: full set of covariates included in every teacher VA regression
global covdesign = "i.year##i.grade##i.subj i.grade##i.subj##c.lag1_math* i.grade##i.subj##c.lag1_read* i.grade##c.sgyts_mean_* i.grade##c.s_mean_* exc_not exc_aig exc_behav exc_educ aigmath aigread disadv lim_eng female black white grade_rep lag1_daysabs lag1_any_discp"

* Excluded/adjustment controls: used in OVB tests (Table A4-A5) and some VAM specs
global covsadj = "pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal"

* Load analysis data (path set via PROJECT_DATA_DIR environment variable)
if c(username)=="shemtov"{
 	global PROJECT_DATA_DIR "/Users/shemtov/Documents/Data_rss"
}

use ${PROJECT_DATA_DIR}/analysis.dta, clear

* Display options
di "Working with design covariates: ${covdesign}"
di "Working with excluded covariates: ${covsadj}"

* Install required packages if not already present
capture which gtools
if _rc {
    ssc install gtools
}

capture which ivreg2
if _rc {
    ssc install ivreg2
}
