clear all
clear matrix
set more off

* Load the residuals
use temp/teach_mean_resids_long.dta, replace

* Merge on the teacher Xs
merge m:1 teachid year using data/teacher_covars.dta, nogen keep(3)

*** 1) Basic regressions
* Code covariates
gen female = gender == "F"
gen non_white = ethnic == "B"
gen birth_year = year(bdate)
gen age = year - birth_year
su age, d
replace age = r(p1) if age < r(p1)
replace age = r(p99) if age < r(p1)
gen higher_ba = educ_lvl_cd >= 5

* test experience cals
sort teachid year
bys teachid (year): gen nprior = _n

* Combined
eststo clear
local c = 0
foreach out of varlist testscores_r studypca_r behave_r  aoc_crim_r aoc_index_r aoc_incar_r {
	qui eststo col`c': reg `out'  female non_white age nprior higher_ba score_normed relative_pay , cluster(teachid)
	qui gdistinct teachid if e(sample) == 1
	qui estadd scalar nteach = r(ndistinct) : col`c'
	local ++ c
}

esttab using tables/tableA1.tex, replace se mtitles("Test scores" "Study skills" "Behaviors" "Criminal arrest" "Index crime" "Incarceration") stats(N nteach, labels("N teacher-years" "N teachers")) coeflabels(female "Female" non_white "Non-white" age "Age" higher_ba "Masters or higher" score_normed "Averge test score" relative_pay "Pay (standardized)" nprior "Prior experience" _cons "Constant")


*** 2) Match effects in race / gender
use data/teacher_covars.dta, clear
keep teachid year gender ethnic
gsort teachid -year
bys teachid: gen keeper = _n == 1
keep if keeper
gen female = gender == "F"
gen non_white = ethnic == "B"
tempfile teacher_demos
save `teacher_demos', replace

* Race
use temp/teach_mean_resids_covwhite.dta, replace

foreach varstub in "testscores_r" "studypca_r" "behavpca_r" "aoc_crim_r" "aoc_index_r" "aoc_incar_r" {
	egen `varstub'0_mean = rowmean(`varstub'0*)
	egen `varstub'1_mean = rowmean(`varstub'1*)
	gen `varstub'_dif = `varstub'1_mean - `varstub'0_mean
}

keep teachid *_dif
merge 1:1 teachid  using `teacher_demos', nogen keep(3)

eststo clear
local c = 0
foreach varstub in "testscores_r" "studypca_r" "behavpca_r" "aoc_crim_r" "aoc_index_r" "aoc_incar_r" {
	qui eststo col`c': reg `varstub'_dif female non_white, cluster(teachid)
	qui gdistinct teachid if e(sample) == 1
	qui estadd scalar nteach = r(ndistinct) : col`c'
	local ++ c
}

esttab using tables/tableA12b.tex, replace se mtitles("Test scores" "Study skills" "Behaviors" "Criminal arrest" "Index crime" "Incarceration") stats(N nteach, labels("N teacher-years" "N teachers")) coeflabels(female "Female" non_white "Non-white" _cons "Constant")

* Gender
use temp/teach_mean_resids_covmale.dta, replace

foreach varstub in "testscores_r" "studypca_r" "behavpca_r" "aoc_crim_r" "aoc_index_r" "aoc_incar_r" {
	egen `varstub'0_mean = rowmean(`varstub'0*)
	egen `varstub'1_mean = rowmean(`varstub'1*)
	gen `varstub'_dif = `varstub'1_mean - `varstub'0_mean
}

keep teachid *_dif
merge 1:1 teachid  using `teacher_demos', nogen keep(3)

eststo clear
local c = 0
foreach varstub in "testscores_r" "studypca_r" "behavpca_r" "aoc_crim_r" "aoc_index_r" "aoc_incar_r" {
	qui eststo col`c': reg `varstub'_dif female non_white, cluster(teachid)
	qui gdistinct teachid if e(sample) == 1
	qui estadd scalar nteach = r(ndistinct) : col`c'
	local ++ c
}

esttab using tables/tableA12a.tex, replace se mtitles("Test scores" "Study skills" "Behaviors" "Criminal arrest" "Index crime" "Incarceration") stats(N nteach, labels("N teacher-years" "N teachers")) coeflabels(female "Female" non_white "Non-white" _cons "Constant")
esttab, se mtitles("Test scores" "Study skills" "Behaviors" "Criminal arrest" "Index crime" "Incarceration") stats(N nteach, labels("N teacher-years" "N teachers")) coeflabels(female "Female" non_white "Non-white" _cons "Constant")


