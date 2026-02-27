clear all
clear matrix
set more off
capture restore

* Load data and options
do code/set_options.do 
do code/preamble.do
global vam_measure = "none"

* Run short-run regressions
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
    eststo iv`mcount': ivreghdfe ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst absorb(school_fe#grade)
    estadd local design_controls = "\checkmark" : iv`mcount'
    estadd local sgfe = "\checkmark" : iv`mcount'
    estadd local ffirst = round(e(first),-1) : iv`mcount'
    test ${lomvar} = 1
    estadd local pvalone = round(r(p),.001) : iv`mcount'
    local mcount = `mcount' + 1

    * School-switchers
    capture drop touse
    gen touse = ebar_school
    eststo iv`mcount': ivreghdfe ${outcome} (${lomvar} = vam_entry_school) any_entry_school ${covdesign}, cluster(mastid) ffirst absorb(school_fe#grade)
    estadd local design_controls = "\checkmark" : iv`mcount'
    estadd local sgfe = "\checkmark" : iv`mcount'
    estadd local ffirst = round(e(first),-1) : iv`mcount'
    test ${lomvar} = 1
    estadd local pvalone = round(r(p),.001) : iv`mcount'
    local mcount = `mcount' + 1

    restore
}

esttab iv1 iv2 iv3 iv4 iv5 iv6 using tables/tableA6.tex, tex replace ///
        keep(${lomvar}) se nostar stats(N r2 design_controls sgfe ffirst pvalone, labels("Observations" "R2" "Design controls" "School-grade FE" "First stage F" "$\Pr(\lambda = 1)$")) ///
        coeflabel(${lomvar} "$\hat{\alpha}_j$") ///
        substitute(\_ _) mlabels("Schl-grd" "Schl" "Schl-grd" "Schl" "Schl-grd" "Schl") ///
        mgroups("Test scores" "Behaviors" "Study skills", pattern(1 0 1 0 1 0)  ///
            prefix(\multicolumn{@span}{c}{) suffix(})   ///
            span erepeat(\cmidrule(lr){@span}))       





