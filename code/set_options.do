* Set universal design options
global covdesign = "i.year##i.grade##i.subj i.grade##i.subj##c.lag1_math* i.grade##i.subj##c.lag1_read* i.grade##c.sgyts_mean_* i.grade##c.s_mean_* exc_not exc_aig exc_behav exc_educ aigmath aigread disadv lim_eng female black white grade_rep lag1_daysabs lag1_any_discp"

global covsadj = "pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal"

* Load analysis data
gzuse data/analysis_data.dta.gz, clear
sample 10

* Display options
di "Working with design covariates: ${covdesign}"
di "Working with excluded covariates: ${covsadj}"

* Vam function
do code/vam.ado

* Package analysis 
capture ssc install gtools
capture ssc install ivreg2
