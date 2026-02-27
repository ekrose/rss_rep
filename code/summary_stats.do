*** Produce summary statitisc
*** 0) Load and prep data and set options
clear all
clear matrix
set more off

*** 1) Build risk measure
* Load data and options
do code/set_options.do

* Execute preamble
do code/preamble.do

qui reg aoc_crim ${covdesign} ${covsadj}
predict arrest_hat, xb
su arrest_hat, d
gen atrisk = arrest_hat >= r(p50)

keep atrisk mastid teachid year subj
tempfile _atrisk
save `_atrisk', replace

***2) Summary statistics
* Load data and options 
do code/set_options.do

* Merge back atrisk
encode subject, gen(subj)
merge 1:1 mastid teachid year subj using `_atrisk', nogen keep(1 3)

* Restrict grades and years
keep if grade > 3
keep if (year >= 1997)

* Keep only teachers that appear in at least 2 years per subject
bys teachid subject: egen nteachhr_years = nvals(year)
drop if nteachhr_years < 2
capture drop nteachhr_years 

* Other covariates
capture gen male = 1 - female
gen college_bound = inlist(bound_for_code,"1","2") if !missing(bound_for_code)
replace class_rank_w = 1 if class_rank_w > 1 & !missing(class_rank_w)

* Fill in twin indicators
gen twins_all = twin_id
replace twins_all = -99 if twins_all == .
bys twin_id year grade: egen sgend_twin = max(female)
bys twin_id year grade: egen sgend_twin2 = min(female)
gen twins_same_gend = twin_id if sgend_twin == sgend_twin2

* Beahvioral pca
foreach var of varlist lead1_daysabs lead1_any_discp lead_grade_rep {
    capture drop _tmp
    bys year grade: egen _tmp = sd(`var')
    gen `var'_norm = `var' / _tmp
}
pca lead1_daysabs_norm lead1_any_discp_norm lead_grade_rep_norm
predict behavpca, score
qui su behavpca
replace behavpca = -behavpca

qui pca homework freeread watchtv
qui predict studypca, score

replace aoc_traff = aoc_traff + aoc_infrac > 0 if aoc_traff != .


**********************************************
* Basic summary stats
**********************************************
* First 2 columns (all sample)
eststo clear
eststo cols1: estpost tabstat male black disadv lim_eng pared_hsorless pared_somecol pared_baormore readscal mathscal daysabs any_discp oss lead_grade_rep behavpca homework freeread watchtv studypca gpa_weighted class_rank_w grad college_bound aoc_any aoc_traff aoc_crim aoc_index aoc_crim_conv aoc_incar aoc_cost_wtp aoc_total_cost_wtp aoc_cost_bottomup aoc_total_cost_bottomup, statistics(mean sd) columns(statistics)
distinct teachid
estadd scalar nteach = r(ndistinct) :cols1
distinct mastid
estadd scalar nstud = r(ndistinct) :cols1
distinct twin_id
estadd scalar ntwins = r(ndistinct) :cols1

* Second 2 columns (sample for which we observe CJC)
eststo cols2: estpost tabstat male black disadv lim_eng pared_hsorless pared_somecol pared_baormore readscal mathscal daysabs any_discp oss lead_grade_rep behavpca homework freeread watchtv studypca gpa_weighted class_rank_w grad college_bound aoc_any aoc_traff aoc_crim aoc_index aoc_crim_conv aoc_incar aoc_cost_wtp aoc_total_cost_wtp aoc_cost_bottomup aoc_total_cost_bottomup if aoc_crim != ., statistics(mean sd) columns(statistics)
distinct teachid if aoc_crim != .
estadd scalar nteach = r(ndistinct) :cols2
distinct mastid if aoc_crim != .
estadd scalar nstud = r(ndistinct) :cols2
distinct twin_id if aoc_crim != .
estadd scalar ntwins = r(ndistinct) :cols2

* Third 2 columns (youth with a CJC)
eststo cols3: estpost tabstat male black disadv lim_eng pared_hsorless pared_somecol pared_baormore readscal mathscal daysabs any_discp oss lead_grade_rep behavpca homework freeread watchtv studypca gpa_weighted class_rank_w grad college_bound aoc_any aoc_traff aoc_crim aoc_index aoc_crim_conv aoc_incar aoc_cost_wtp aoc_total_cost_wtp aoc_cost_bottomup aoc_total_cost_bottomup if aoc_crim == 1, statistics(mean sd) columns(statistics)
distinct teachid if aoc_crim == 1
estadd scalar nteach = r(ndistinct) :cols3
distinct mastid if aoc_crim == 1
estadd scalar nstud = r(ndistinct) :cols3
distinct twin_id if aoc_crim == 1
estadd scalar ntwins = r(ndistinct) :cols3

esttab cols1 cols2 cols3 using tables/table1.tex, tex replace cells("mean(fmt(a2)) sd(fmt(a2))") nomtitles nonumber stats(N nteach nstud ntwins, labels("N student-subject-years" "N teachers" "N students" "N twin pairs")) coeflabels(male "\textbf{Demographics}\\ \ Male" black "\ Black" disadv "\ Receives free / subsidized lunch" lim_eng "\ Limited English" pared_hsorless "\ Parents have HS education or less" pared_somecol "\ Parents have some college" pared_baormore "\ Parents have 4-year degree"   readscal "\\ \textbf{Short-run outcomes}\\ \ Standardized reading scores" mathscal "\ Standardized math scores" daysabs "\ Days absent" any_discp "\ Any discipline" oss "\ Any out-of-school suspension" lead_grade_rep "\ Repeat grade" behavpca "\ Behavioral index" homework "\ Time spent on homework" freeread "\ Time spent reading" watchtv "\ Time spent watching TV" studypca "\ Study skills index" gpa_weighted "\\ \textbf{Long-run outcomes}\\ \ 12th grade GPA (0-6 scale)" class_rank_w "\ 12th grade class rank" grad "Graduate high school" college_bound "\ Plans to attend 4-year college" aoc_any "\ Any CJC 16-21" aoc_traff "\ Traffic infraction" aoc_crim "\ Criminal arrest" aoc_index "\ Index crime arrest" aoc_crim_conv "\ Criminal conviction" aoc_incar "\ Incarcerated") mgroups("Full sample" "Sample for which we observe CJC" "Youth with a criminal arrest", pattern( 1 1 1 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) 


**********************************************
* Heterogeneity summary statistics
**********************************************
* Define 4 primary heterogeneity sub-samples


* Main sample
eststo clear
eststo cols1: estpost tabstat male black disadv lim_eng pared_hsorless pared_somecol pared_baormore readscal mathscal daysabs any_discp oss lead_grade_rep behavpca homework freeread watchtv studypca gpa_weighted class_rank_w grad college_bound aoc_any aoc_traff aoc_crim aoc_index aoc_crim_conv aoc_incar, statistics(mean) columns(statistics)
distinct teachid
estadd scalar nteach = r(ndistinct) :cols1
distinct mastid
estadd scalar nstud = r(ndistinct) :cols1
distinct twin_id
estadd scalar ntwins = r(ndistinct) :cols1

* Sub-samples
local cc = 2
foreach var in white male disadv atrisk{
    foreach l in 1 0{
        eststo cols`cc': estpost tabstat male black disadv lim_eng pared_hsorless pared_somecol pared_baormore readscal mathscal daysabs any_discp oss lead_grade_rep behavpca homework freeread watchtv studypca gpa_weighted class_rank_w grad college_bound aoc_any aoc_traff aoc_crim aoc_index aoc_crim_conv aoc_incar if `var' == `l', statistics(mean) columns(statistics)
        distinct teachid if `var' == `l'
        estadd scalar nteach = r(ndistinct) :cols`cc'
        distinct mastid if `var' == `l'
        estadd scalar nstud = r(ndistinct) :cols`cc'
        distinct twin_id if `var' == `l'
        estadd scalar ntwins = r(ndistinct) :cols`cc'  

        local ++cc
    }
}

esttab cols* using tables/tableA11.tex, tex replace cells("mean(fmt(a2))")  mtitles(" " "White" "Non-White" "Boys" "Girls" "Yes" "No" "High" "Low" ) stats(N nteach nstud ntwins, labels("N student-subject-years" "N teachers" "N students" "N twin pairs")) coeflabels(male "\textbf{Demographics}\\ \ Male" black "\ Black" disadv "\ Receives free / subsidized lunch" lim_eng "\ Limited English" pared_hsorless "\ Parents have HS education or less" pared_somecol "\ Parents have some college" pared_baormore "\ Parents have 4-year degree"   readscal "\\ \textbf{Short-run outcomes}\\ \ Standardized reading scores" mathscal "\ Standardized math scores" daysabs "\ Days absent" any_discp "\ Any discipline" oss "\ Any out-of-school suspension" lead_grade_rep "\ Repeat grade" behavpca "\ Behavioral index" homework "\ Time spent on homework" freeread "\ Time spent reading" watchtv "\ Time spent watching TV" studypca "\ Study skills index" gpa_weighted "\\ \textbf{Long-run outcomes}\\ \ 12th grade GPA (0-6 scale)" class_rank_w "\ 12th grade class rank" grad "Graduate high school" college_bound "\ Plans to attend 4-year college" aoc_any "\ Any CJC 16-21" aoc_traff "\ Traffic infraction" aoc_crim "\ Criminal arrest" aoc_index "\ Index crime arrest" aoc_crim_conv "\ Criminal conviction" aoc_incar "\ Incarcerated") mgroups("Full sample" "Race" "Sex" "Econ. disadv." "Arrest risk", pattern(1 1 0 1 0 1 0 1 0 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) 




