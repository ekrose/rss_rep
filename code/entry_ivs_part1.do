* =============================================================================
* entry_ivs_part1.do — Table 5 (Forecast Unbiasedness IV Tests)
*
* Tests whether teacher VA estimates are forecast-unbiased using an
* instrumental variables strategy based on teacher entry into new schools
* or school-grade combinations. The instrument is the mean leave-out VA
* of teachers entering a school-grade cell for the first time, which is
* plausibly exogenous to student sorting within the receiving school.
*
* For each outcome, the procedure:
*   1. Computes leave-out teacher VA (via fun_vam with vam_measure="none")
*   2. Constructs the entry IV: mean LOO-VA of teachers new to the
*      school-grade (or school), conditional on >= 3 years of school-grade data
*   3. Runs 2SLS: outcome = lambda * ebar_schgrade, instrumented by entry IV
*   4. Tests H0: lambda = 1 (forecast unbiasedness)
*
* Table 5a: short-run outcomes (test scores, behaviors, study skills)
* Table 5b: long-run outcomes (criminal arrest, incarceration, college-bound)
* =============================================================================

*** 0) Load and prep data and set options
clear all
clear matrix
set more off
capture restore
capture log close

* Load data and options
do code/set_options.do
do code/preamble.do
global vam_measure = "none"

*** Short-run outcome entry IVs (Table 5a)
dis "Working on short-run outcomes"
eststo clear
local mcount = 1
foreach outcome of varlist testscores behavpca studypca {
	preserve 
	global outcome "`outcome'"
	fun_vam "${outcome}" 0 

	*** 1) IV tests of forecast bias
	* Entry of new teacher into school grade
	bys school_fe grade year: egen vam_entry_school_grade = mean(loo_ebar_schgrade) if nyears_teach_schgrd == 0 & nyears_schgrd >= 3
	capture drop _tmp
	gen _tmp = vam_entry_school_grade != .
	bys school_fe grade year: egen any_entry_school_grade = max(_tmp)
	drop _tmp
	replace vam_entry_school_grade = 0 if vam_entry_school_grade == .

	* Entry of teacher into school
	bys school_fe grade year: egen vam_entry_school = mean(loo_ebar_school) if nyears_teach_school == 0 & nyears_schgrd >= 3
	gen _tmp = vam_entry_school != .
	bys school_fe grade year: egen any_entry_school = max(_tmp)
	drop _tmp
	replace vam_entry_school = 0 if vam_entry_school == .

	* Endogenous var
	global lomvar "touse"

	* School-grade switchers
	capture drop touse
	gen touse = ebar_schgrade

	* School-grade switchers
	eststo desiv`mcount': ivreg2 ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst 
	estadd local design_controls = "\checkmark" : desiv`mcount'
	estadd local sgfe = "" : desiv`mcount'
	estadd local ffirst = round(el(e(first),4,1),-1) : desiv`mcount'
	test ${lomvar} = 1
	estadd local pvalone = round(r(p),.001) : desiv`mcount'
	local mcount = `mcount' + 1

	* School-switchers
	capture drop touse
	gen touse = ebar_school

	eststo desiv`mcount': ivreg2 ${outcome} (${lomvar} = vam_entry_school) any_entry_school ${covdesign}, cluster(mastid) ffirst
	estadd local design_controls = "\checkmark" : desiv`mcount'
	estadd local sgfe = "" : desiv`mcount'
	estadd local ffirst = round(el(e(first),4,1),-1) : desiv`mcount'
	test ${lomvar} = 1
	estadd local pvalone = round(r(p),.001) : desiv`mcount'
	local mcount = `mcount' + 1

	restore
}    

    esttab desiv1 desiv2 desiv3 desiv4 desiv5 desiv6 using tables/table5a, tex replace ///
            keep(${lomvar}) se nostar stats(N r2 design_controls sgfe ffirst pvalone, labels("Observations" "R2" "Design controls" "School-grade FE" "First stage F" "$\Pr(\lambda = 1)$")) ///
            coeflabel(${lomvar} "$\hat{\alpha}_j$") ///
            substitute(\_ _) mlabels("Schl-grd" "Schl" "Schl-grd" "Schl" "Schl-grd" "Schl") ///
            mgroups("Test scores" "Behaviors" "Study skills", pattern(1 0 1 0 1 0)  ///
                prefix(\multicolumn{@span}{c}{) suffix(})   ///
                span erepeat(\cmidrule(lr){@span}))       

*** Long-run outcome entry IVs (Table 5b)
dis "Working on long-run outcomes"
eststo clear
local mcount = 1
foreach outcome of varlist aoc_crim aoc_incar college_bound {
	preserve 
	global outcome "`outcome'"
	fun_vam "${outcome}" 0 

	*** 1) IV tests of forecast bias
	* Entry of new teacher into school grade
	bys school_fe grade year: egen vam_entry_school_grade = mean(loo_ebar_schgrade) if nyears_teach_schgrd == 0 & nyears_schgrd >= 3
	capture drop _tmp
	gen _tmp = vam_entry_school_grade != .
	bys school_fe grade year: egen any_entry_school_grade = max(_tmp)
	drop _tmp
	replace vam_entry_school_grade = 0 if vam_entry_school_grade == .

	* Entry of teacher into school
	bys school_fe grade year: egen vam_entry_school = mean(loo_ebar_school) if nyears_teach_school == 0 & nyears_schgrd >= 3
	gen _tmp = vam_entry_school != .
	bys school_fe grade year: egen any_entry_school = max(_tmp)
	drop _tmp
	replace vam_entry_school = 0 if vam_entry_school == .

	* Endogenous var
	global lomvar "touse"

	* School-grade switchers
	capture drop touse
	gen touse = ebar_schgrade

	* School-grade switchers
	eststo desiv`mcount': ivreg2 ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst 
	estadd local design_controls = "\checkmark" : desiv`mcount'
	estadd local sgfe = "" : desiv`mcount'
	estadd local ffirst = round(el(e(first),4,1),-1) : desiv`mcount'
	test ${lomvar} = 1
	estadd local pvalone = round(r(p),.001) : desiv`mcount'
	local mcount = `mcount' + 1

	* School-switchers
	capture drop touse
	gen touse = ebar_school

	eststo desiv`mcount': ivreg2 ${outcome} (${lomvar} = vam_entry_school) any_entry_school ${covdesign}, cluster(mastid) ffirst
	estadd local design_controls = "\checkmark" : desiv`mcount'
	estadd local sgfe = "" : desiv`mcount'
	estadd local ffirst = round(el(e(first),4,1),-1) : desiv`mcount'
	test ${lomvar} = 1
	estadd local pvalone = round(r(p),.001) : desiv`mcount'
	local mcount = `mcount' + 1

	restore
}

    esttab desiv1 desiv2 desiv3 desiv4 desiv5 desiv6 using tables/table5b.tex, tex replace ///
            keep(${lomvar}) se nostar stats(N r2 design_controls sgfe ffirst pvalone, labels("Observations" "R2" "Design controls" "School-grade FE" "First stage F" "$\Pr(\lambda = 1)$")) ///
            coeflabel(${lomvar} "$\hat{\alpha}_j$") ///
            substitute(\_ _) mlabels("Schl-grd" "Schl" "Schl-grd" "Schl" "Schl-grd" "Schl") ///
            mgroups("Criminal arrest" "Incarceration" "College bound", pattern(1 0 1 0 1 0)  ///
                prefix(\multicolumn{@span}{c}{) suffix(})   ///
                span erepeat(\cmidrule(lr){@span}))       





