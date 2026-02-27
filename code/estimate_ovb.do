*** 0) Load and prep data and set options
clear all
clear matrix
set more off

* Load data and options
do code/set_options.do 
do code/preamble.do

* RHS variables (must be defined below)
global regvar "teach_fe"

*** 1) OVB tests for short-run outcomes
eststo clear
foreach outcome of varlist testscores behavpca studypca {
    capture drop xhat xhatd teach_fe
    qui areg `outcome' ${covdesign}, abs(teachid)
    local origr2 = e(r2)
    predict xhat, xb
    predict xhatd, xbd
    gen teach_fe = xhatd - xhat
	
	if "`outcome'" == "testscores" {
		local figtitle "2a"
	}
	if "`outcome'" == "behavpca" {
		local figtitle "2b"
	}

    * Regression with all omitted
    capture drop twin_fe covadj_predict
    eststo: reghdfe `outcome' ${covsadj} ${covdesign}, absorb(teachid twin_fe=twins_all#grade#year, savefe)
    estadd local design_controls = "\checkmark"
    estadd local twin_fe = "\checkmark"
    estadd local orig_r2 = round(`origr2',0.0001)
    capture drop covadj_predict
    mat b=e(b)
    mat b=b[1,1..2]
    mat score covadj_predict=b if e(sample) == 1
    replace covadj_predict = covadj_predict + twin_fe

    eststo: reg covadj_predict ${regvar} ${covdesign}, cluster(mastid)
    estadd local design_controls = "\checkmark"
    local slopeCoef = round(_b[${regvar}], 0.0001)
    local slopeSE = round(_se[${regvar}], 0.0001)
    qui su covadj_predict if e(sample) == 1
    replace covadj_predict = covadj_predict - r(mean)

	if inlist("`outcome'","testscores","behavpca") {
		binscatter covadj_predict ${regvar}, controls(${covdesign}) xtitle("Estimated teacher effect") ytitle("Omitted variables fit") yscale(range(-0.03 0.03)) ylabel(-0.03(0.01)0.03) graphregion(color(white)) legend(off) note("Slope: `slopeCoef' (`slopeSE')")
		graph export figures/figure`figtitle'.pdf, replace
	}
}

esttab, keep(pared* lag2_mathscal lag2_readscal ${regvar}) stats(N r2 orig_r2 design_controls twin_fe, labels("Observations" "R2" "Original R2" "Design controls" "Twin FE")) 
esttab using tables/tableA4.tex, tex replace ///
        keep(pared* lag2_mathscal lag2_readscal ${regvar}) se nostar stats(N r2 orig_r2 design_controls twin_fe, labels("Observations" "R2" "Original R2" "Design controls" "Twin FE")) ///
        coeflabel(pared_nohs "No high school" pared_hsorless "High school only" pared_somecol "Some college" pared_baormore "BA or more" lag2_mathscal "Lag 2 math" lag2_readscal "Lag 2 reading" ${regvar} "Teacher effect") ///
        substitute(\_ _) mlabels("$Y$" "$\hat{Y}$" "$Y$" "$\hat{Y}$" "$Y$" "$\hat{Y}$") ///
        mgroups("Test scores" "Behvioral index" "Study skills index", pattern(1 0 1 0 1 0)  ///
            prefix(\multicolumn{@span}{c}{) suffix(})   ///
            span erepeat(\cmidrule(lr){@span}))   

*** 2) OVB tests for long-run academic outcomes
eststo clear
foreach outcome of varlist college_bound gpa_weighted {
    capture drop xhat xhatd teach_fe
    qui areg `outcome' ${covdesign}, abs(teachid)
    local origr2 = e(r2)
    predict xhat, xb
    predict xhatd, xbd
    gen teach_fe = xhatd - xhat

	
	if "`outcome'" == "college_bound" {
		local figtitle "A3c"
	}
	if "`outcome'" == "gpa_weighted" {
		local figtitle "A3d"
	}

    * Regression with all omitted
    capture drop twin_fe covadj_predict
    eststo: reghdfe `outcome' ${covsadj} ${covdesign}, absorb(teachid twin_fe=twins_all#grade#year, savefe)
    estadd local design_controls = "\checkmark"
    estadd local twin_fe = "\checkmark"
    estadd local orig_r2 = round(`origr2',0.0001)
    capture drop covadj_predict
    mat b=e(b)
    mat b=b[1,1..2]
    mat score covadj_predict=b if e(sample) == 1
    replace covadj_predict = covadj_predict + twin_fe

    eststo: reg covadj_predict ${regvar} ${covdesign}, cluster(mastid)
    estadd local design_controls = "\checkmark"
    local slopeCoef = round(_b[${regvar}], 0.0001)
    local slopeSE = round(_se[${regvar}], 0.0001)
    qui su covadj_predict if e(sample) == 1
    replace covadj_predict = covadj_predict - r(mean)

	binscatter covadj_predict ${regvar}, controls(${covdesign}) xtitle("Estimated teacher effect") ytitle("Omitted variables fit") yscale(range(-0.025 0.025)) ylabel(-0.025(0.005)0.025) graphregion(color(white)) legend(off) note("Slope: `slopeCoef' (`slopeSE')")
	graph export figures/figure`figtitle'.pdf, replace
}

*** 3) OVB tests for CJC long-run outcomes
eststo clear
foreach outcome of varlist aoc_any aoc_crim aoc_index aoc_incar {
    capture drop xhat xhatd teach_fe
    qui areg `outcome' ${covdesign}, abs(teachid)
    local origr2 = e(r2)
    predict xhat, xb
    predict xhatd, xbd
    gen teach_fe = xhatd - xhat

	if "`outcome'" == "aoc_any" {
		local figtitle "A3a"
	}
	if "`outcome'" == "aoc_index" {
		local figtitle "A3b"
	}

    * Regression with all omitted
    capture drop twin_fe covadj_predict
    eststo: reghdfe `outcome' ${covsadj} ${covdesign}, absorb(teachid twin_fe=twins_all#grade#year, savefe)
    estadd local design_controls = "\checkmark"
    estadd local twin_fe = "\checkmark"
    estadd local orig_r2 = round(`origr2',0.0001)
    capture drop covadj_predict
    mat b=e(b)
    mat b=b[1,1..2]
    mat score covadj_predict=b if e(sample) == 1
    replace covadj_predict = covadj_predict + twin_fe

    eststo: reg covadj_predict ${regvar} ${covdesign}, cluster(mastid)
    estadd local design_controls = "\checkmark"
    local slopeCoef = round(_b[${regvar}], 0.0001)
    local slopeSE = round(_se[${regvar}], 0.0001)
    qui su covadj_predict if e(sample) == 1
    replace covadj_predict = covadj_predict - r(mean)

	if inlist("`outcome'","aoc_any","aoc_index") {
		binscatter covadj_predict ${regvar}, controls(${covdesign}) xtitle("Estimated teacher effect") ytitle("Omitted variables fit") yscale(range(-0.025 0.025)) ylabel(-0.025(0.005)0.025) graphregion(color(white)) legend(off) note("Slope: `slopeCoef' (`slopeSE')")
		graph export figures/figure`figtitle'.pdf, replace
	}
}

esttab, keep(pared* lag2_mathscal lag2_readscal ${regvar}) stats(N r2 orig_r2 design_controls twin_fe, labels("Observations" "R2" "Original R2" "Design controls" "Twin FE")) 
esttab using tables/tableA5.tex, tex replace ///
        keep(pared* lag2_mathscal lag2_readscal ${regvar}) se nostar stats(N r2 orig_r2 design_controls twin_fe, labels("Observations" "R2" "Original R2" "Design controls" "Twin FE")) ///
        coeflabel(pared_nohs "No high school" pared_hsorless "High school only" pared_somecol "Some college" pared_baormore "BA or more" lag2_mathscal "Lag 2 math" lag2_readscal "Lag 2 reading" ${regvar} "Teacher effect") ///
        substitute(\_ _) mlabels("$Y$" "$\hat{Y}$" "$Y$" "$\hat{Y}$" "$Y$" "$\hat{Y}$") ///
        mgroups("Any arrest" "Criminal arrest" "Index crime" "Incarceration", pattern(1 0 1 0 1 0 1 0)  ///
            prefix(\multicolumn{@span}{c}{) suffix(})   ///
            span erepeat(\cmidrule(lr){@span}))   

