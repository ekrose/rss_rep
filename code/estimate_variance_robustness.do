clear all
clear matrix
set more off
capture ssc install tuples
set seed 617238

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
gzuse data/analysis_data.dta.gz, clear

* Execute preamble
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


