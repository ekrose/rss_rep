clear all
clear matrix
set more off

* Load data and options 
do code/set_options.do 
do code/preamble.do

* Hetereogenous x (must be binary)
global w "`2'"

if "${w}" == "atrisk" {
    qui reg aoc_crim ${covdesign} ${covsadj}
    predict arrest_hat
    su arrest_hat, d
    gen atrisk = arrest_hat >= r(p50)
}

if "${w}" == "lowscores" {
    gen lowscores = testscores < 0
}

if "${w}" == "racegend" {
    gen racegend = 0 if black == 0 & male == 0
    replace racegend = 1 if black == 1 & male == 0
    replace racegend = 3 if black == 0 & male == 1
    replace racegend = 4 if black == 1 & male == 1
}

drop if ${w} == .

* Residuals
foreach var of varlist aoc_any aoc_crim aoc_index aoc_incar aoc_traff testscores behavpca studypca gpa_weighted college_bound grad {
    reghdfe `var' ${covdesign}, abs(${w}#teachid) resid
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

