clear all
clear matrix
set more off
capture restore

* Load data and options
do set_options.do
do preamble.do
global outcome "aoc_crim"

egen district = group(lea)
fun_vam "${outcome}" 0 

eststo clear
local mcount = 1

* Entry of new teacher into school grade
capture drop _tmp
bys school_fe grade year: egen vam_entry_school_grade = mean(loo_ebar_schgrade) if nyears_teach_schgrd == 0 & nyears_schgrd >= 3
gen _tmp = vam_entry_school_grade != .
bys school_fe grade year: egen any_entry_school_grade = max(_tmp)
capture drop _tmp
replace vam_entry_school_grade = 0 if vam_entry_school_grade == .

* Endogenous var
global lomvar "touse"

** School-grade switchers
capture drop touse
gen touse = ebar_schgrade
eststo iv`mcount': ivreg2 ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst
estadd local design_controls = "\checkmark" : iv`mcount'
estadd local ffirst = round(e(first),-1) : iv`mcount'
test ${lomvar} = 1
estadd local pvalone = round(r(p),.001) : iv`mcount'
local mcount = `mcount' + 1

eststo iv`mcount': ivreghdfe ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst absorb(school_fe#grade)
estadd local design_controls = "\checkmark" : iv`mcount'
estadd local sgfe = "\checkmark" : iv`mcount'
estadd local ffirst = round(e(first),-1) : iv`mcount'
test ${lomvar} = 1
estadd local pvalone = round(r(p),.001) : iv`mcount'
local mcount = `mcount' + 1

eststo iv`mcount': ivreghdfe ${outcome} (${lomvar} = vam_entry_school_grade) any_entry_school_grade ${covdesign}, cluster(mastid) ffirst absorb(school_fe#grade district#grade#year)
estadd local design_controls = "\checkmark" : iv`mcount'
estadd local sgfe = "\checkmark" : iv`mcount'
estadd local distyear = "\checkmark" : iv`mcount'
estadd local ffirst = round(e(first),-1) : iv`mcount'
test ${lomvar} = 1
estadd local pvalone = round(r(p),.001) : iv`mcount'
local mcount = `mcount' + 1

eststo iv`mcount': reg all_${outcome}_idx vam_entry_school_grade any_entry_school_grade ${covdesign}, cluster(mastid)
estadd local design_controls = "\checkmark" : iv`mcount'
local mcount = `mcount' + 1

eststo iv`mcount': reghdfe all_${outcome}_idx vam_entry_school_grade any_entry_school_grade ${covdesign}, cluster(mastid) absorb(school_fe#grade)
estadd local design_controls = "\checkmark" : iv`mcount'
estadd local sgfe = "\checkmark" : iv`mcount'
local mcount = `mcount' + 1

eststo iv`mcount': reghdfe all_${outcome}_idx vam_entry_school_grade any_entry_school_grade ${covdesign}, cluster(mastid) absorb(school_fe#grade district#grade#year)
estadd local design_controls = "\checkmark" : iv`mcount'
estadd local sgfe = "\checkmark" : iv`mcount'
estadd local distyear = "\checkmark" : iv`mcount'
local mcount = `mcount' + 1

esttab iv1 iv2 iv3 iv4 iv5 iv6 using tables/tableA7.tex, tex replace ///
        keep(${lomvar} vam_entry_school_grade) se nostar stats(N r2 design_controls sgfe distyear ffirst pvalone, labels("Observations" "R2" "Design controls" "School-grade FE" "Dist-grade-year FE" "First stage F" "\$Pr(\lambda = 1)$")) ///
        coeflabel(${lomvar} "$\hat{\alpha}_j$" vam_entry_school_grade "$Z_{it}$") ///
        substitute(\_ _) nomtitles ///
        mgroups("Outcome: $Y$" "Outcome: $\hat{Y}_{excluded}$", pattern(1 0 0 1 0 0)  ///
            prefix(\multicolumn{@span}{c}{) suffix(})   ///
            span erepeat(\cmidrule(lr){@span}))   


