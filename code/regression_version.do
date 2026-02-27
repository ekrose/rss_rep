*** 0) Load and prep data and set options
clear all
clear mata
clear matrix
set more off

* Load data and options
do code/set_options.do

* Execute preamble
do code/preamble.do

*** 1) Estimate regression of shrunken VA on long-run outcomes
* Separate tables
foreach shortrun of varlist testscores behavpca {
    eststo clear
    preserve
    * Add VAM measures
	capture program drop vam
	clear mata
	do code/vam.ado
	capture log close
    global vam_measure = "chetty"
    fun_vam "`shortrun'" 0 

    if "`shortrun'" == "testscores"{
        local outlabel "Test score VA"
        local tabtitle "A2"
    }
    else if "`shortrun'" == "behavpca"{
        local outlabel "Behavioral index VA"
        local tabtitle "A3"
    }

    * Regress on outcomes
    foreach longrun of varlist aoc_any aoc_crim aoc_index aoc_incar gpa_weighted grad college_bound {
        * Run reg
        eststo: reghdfe `longrun' vam ${covdesign}, vce(cluster teachid mastid) noabsorb
        estadd local design_controls = "\checkmark"
        local beta = _b[vam]
        su vam if e(sample)
        estadd local sdeffect = round(`beta'*r(sd), 0.0001)
    }

    esttab using tables/table`tabtitle', tex replace keep(vam) coeflabel(vam "`outlabel'") ///
        se nostar stats(design_controls sdeffect r2 N, labels("Design controls" "1SD effect" "R2" "Observations")) ///
        mlabels("Any CJC" "Criminal arrest" "Index crime" "Incarceration" "12th grade GPA" "Graduation" "College bound" ) substitute(\_ _) ///
        mgroups("CJC outcomes" "Academic outcomes", pattern(1 0 0 0 1 0 0)  ///
            prefix(\multicolumn{@span}{c}{) suffix(})   ///
            span erepeat(\cmidrule(lr){@span}))
    restore
}
