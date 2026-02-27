*** Estimation Residuals for u-stat estimation of variance covariance stuff
*** 0) Load and prep data and set options
clear all
clear matrix
set more off

* Outcome to study
global outcome "`1'"

* Load data and options (spec, drop pval are options)
do set_options.do `2' `3'
di "Working with spec ${spec} and dropping low pvalues ${droppval}"

* Execute preamble
do preamble.do

* Get the residuals
foreach var of varlist aoc_cost_wtp aoc_total_cost_wtp aoc_cost_bottomup aoc_total_cost_bottomup  aoc_any aoc_crim aoc_traff aoc_index aoc_incar gpa_weighted class_rank_w college_bound grad {
    areg `var' $covdesign, abs(teachid)   
    predict `var'_r, dresiduals 
}

foreach var of varlist behavpca_normalized daysabs oss lead1_daysabs lead_grade_rep lead1_oss lead1_iss lead1_detention {
    areg `var' $covdesign, abs(teachid)   
    predict `var'_r, dresiduals 
}

foreach prefix in "" "lead1_" "lead2_" "lead3_" "lead4_" {
    foreach var of varlist testscores any_discp studypca {
        areg `prefix'`var' $covdesign, abs(teachid)    
        predict `prefix'`var'_r, dresiduals 
    }

    areg `prefix'mathscal $covdesign if subject == "math" | subject == "hr", abs(teachid)    
    predict `prefix'math_r, dresiduals 

    areg `prefix'readscal $covdesign if subject == "eng" | subject == "hr", abs(teachid)    
    predict `prefix'eng_r, dresiduals 

    if "`prefix'" != "lead4_" {
        areg `prefix'behavpca $covdesign, abs(teachid)    
        predict `prefix'behave_r, dresiduals 
    }
}

* Check that there are no missing values in the design matrix, i.e., in "$covdesign"
assert missing(testscores) == missing(testscores_r)


* Collapse by teacher year
preserve
collapse (mean) *_r (max) school_fe = school_fe (count) obs_cog = testscores_r obs_crime = aoc_crim_r, by(teachid year)

sort teachid year
by teachid: gen obs = _n

* Save long version for KSS-routines
save ../dump/teach_mean_resids_spec${spec}_droppval${droppval}_long_samp100.dta, replace

* Save wide version for U-stat estimators
drop year 
* reshape wide *_r, i(teachid school_fe) j(obs)
reshape wide *_r obs_* school_fe, i(teachid) j(obs)
save ../dump/teach_mean_resids_spec${spec}_droppval${droppval}_samp100.dta, replace
restore

* Collapse by teacher-school
preserve
collapse (mean) *_r, by(teachid school_fe)

sort teachid school_fe
by teachid: gen obs = _n

drop school_fe
reshape wide *_r, i(teachid) j(obs)

save ../dump/teachSchl_mean_resids_spec${spec}_droppval${droppval}_samp100.dta, replace
restore

* Separately by grade
foreach grade of numlist 4/8 {
    preserve
    collapse (mean) *_r if grade == `grade', by(teachid year)

    sort teachid year
    by teachid: gen obs = _n

    drop year
    reshape wide *_r, i(teachid) j(obs)

    save ../dump/teach_mean_resids_spec${spec}_droppval${droppval}_gr`grade'_samp100.dta, replace
    restore
}

* Keep only years with valid crime and cog residuals
preserve
collapse (mean) *_r, by(teachid year)
keep if aoc_any_r != . & testscores_r != .

sort teachid year
by teachid: gen obs = _n
    * max obs is 11

drop year
reshape wide *_r, i(teachid) j(obs)

save ../dump/teach_mean_resids_spec${spec}_droppval${droppval}_cog_crime_non_missing_samp100.dta, replace
restore

