*** Estimation Residuals for u-stat estimation of variance covariance stuff
*** 0) Load and prep data and set options
clear all
clear matrix
set more off

* Directory path globals
global filepath "/accounts/projects/crwalters/cncrime/teachers_final"
local wm_data "${filepath_wm}/data"

local droppval = 0

dis "`0'"

if 0 {
    * Potential options to try
    global base1 = "i.year##i.grade##i.subj i.grade##i.subj##c.lag1_math* i.grade##i.subj##c.lag1_read*"
    global base2 = "i.year##i.grade##i.subj i.grade##c.lag1_math* i.grade##c.lag1_read*"
    global sgym = "i.grade##c.sgy_mean_*"
    global cmean = "i.grade##c.sgyts_mean_*"
    global smean = "i.grade##c.s_mean_*"
    global covs = "lag1_daysabs lag1_any_discp exc_not exc_aig exc_behav exc_educ aigmath aigread disadv lim_eng female black white grade_rep pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal"
    local num_cov : list sizeof global(covs)
    di `num_cov'

    * Basic covariates
    local iter = 0
    tuples base1 base2 sgym cmean smean, asis conditionals((1|2) !(1&2) (3|4|5))
    forvalues i = 1/`ntuples' {
        global covbase = ""
        tuples base1 base2 sgym cmean smean, asis conditionals((1|2) !(1&2) (3|4|5))
        foreach a of local tuple`i' {
            global covbase =  "${covbase} ${`a'}"
        }
        * Add random subset k other covariates
        forvalues ncov = 1/`num_cov' {
            tuples lag1_daysabs lag1_any_discp exc_not exc_aig exc_behav exc_educ aigmath aigread disadv lim_eng female black white grade_rep pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal, min(`ncov') max(`ncov')
            local totry = min(3, `ntuples')
            forvalues l = 1/`totry' {
                local todo = floor(runiform()*`ntuples'+1)
                global covdesign = "${covbase} `tuple`todo''"
                di "${covdesign}"
            } 
        }
    }
}
else {
    * Options
    global outcome = "testscores"
    global covdesign = "`0'"
    global covdesign = regexr("$covdesign", "^[0-9]+","")
    di "Working with covdesign ${covdesign}"
    dis "-------------"

    global covsadj = "pared_nohs pared_hsorless pared_somecol pared_baormore lag2_mathscal lag2_readscal"
    local iter = `1'
    di "Working on iteration `iter'"

    * Load data
    * use /scratch/public/ncrime/analysis_01_16_2021.dta, clear
    * use /scratch/public/ncrime/analysis_08_03_2021.dta, clear
    use /scratch/public/ncrime/analysis_11_01_2021.dta, clear

    * Drop low pvalues if desired
    if `droppval' == 1 {
        drop if pval < 0.1
    }

    * Execute preamble
    do ${filepath}/code/preamble.do

    * Get the residuals
    * foreach var of varlist aoc_crim aoc_incar grad college_bound testscores studypca behavpca {
    foreach var of varlist aoc_crim college_bound testscores behavpca {
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
    if `droppval' == 1 {
        save /scratch/public/ncrime/robust/resids_iter`iter'.dta, replace
    }

    if `droppval' == 0 {
        save /scratch/public/ncrime/robust/resids_NOdroppval_iter`iter'.dta, replace
    }
}


