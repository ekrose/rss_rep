* =============================================================================
* estimate_variance.do — Tables 2-3, Figure A1 (Stata residuals step)
*
* Regresses each outcome on design covariates absorbing teacher FE, extracts
* teacher-year mean residuals ("dresiduals" = residual + absorbed teacher FE),
* and saves them in several formats used by subsequent Python U-stat scripts:
*
* Output files:
*   temp/teach_mean_resids_long.dta — long format (one row per teacher-year),
*       used by vcov_t_sensitivity.py and teacher_chars.do
*   temp/teach_mean_resids.dta — wide format (one row per teacher, columns
*       indexed by observation number 1..T_j), used by most Python scripts
*   temp/teachSchl_mean_resids.dta — wide format collapsed by teacher-school
*       (for within-school analyses)
*   temp/teach_mean_resids_gr{4..8}.dta — wide format by grade
*   temp/teach_mean_resids_cog_crime_non_missing.dta — wide format restricted
*       to teacher-years with non-missing cognitive and crime residuals
*
* Outcomes include: short-run (test scores, behaviors, study skills, absences,
* suspensions, etc.) at current and lead periods, and long-run (criminal
* justice, GPA, graduation, college attendance).
* =============================================================================
clear all
clear matrix
set more off

* Load data and options
do code/set_options.do
do code/preamble.do

* Regress each outcome on design covariates, absorbing teacher FE.
* "dresiduals" = residual + absorbed teacher effect, so teacher-year means
* of dresiduals estimate teacher-year effects net of covariates.
foreach var of varlist  aoc_any aoc_crim aoc_traff aoc_index aoc_incar gpa_weighted class_rank_w college_bound grad {
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


* Collapse student-level dresiduals to teacher-year means.
* Also carry forward school_fe (max within teacher-year) and observation counts.
preserve
collapse (mean) *_r (max) school_fe = school_fe (count) obs_cog = testscores_r obs_crime = aoc_crim_r, by(teachid year)

sort teachid year
by teachid: gen obs = _n

* Save long version for KSS-routines
save temp/teach_mean_resids_long.dta, replace

* Save wide version for U-stat estimators
drop year 
reshape wide *_r obs_* school_fe, i(teachid) j(obs)
save temp/teach_mean_resids.dta, replace
restore

* Collapse by teacher-school (for within-school variance decomposition, Table A9)
preserve
collapse (mean) *_r, by(teachid school_fe)

sort teachid school_fe
by teachid: gen obs = _n

drop school_fe
reshape wide *_r, i(teachid) j(obs)

save temp/teachSchl_mean_resids.dta, replace
restore

* Separately by grade
foreach grade of numlist 4/8 {
    preserve
    collapse (mean) *_r if grade == `grade', by(teachid year)

    sort teachid year
    by teachid: gen obs = _n

    drop year
    reshape wide *_r, i(teachid) j(obs)

    save temp/teach_mean_resids_gr`grade'.dta, replace
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

save temp/teach_mean_resids_cog_crime_non_missing.dta, replace
restore

