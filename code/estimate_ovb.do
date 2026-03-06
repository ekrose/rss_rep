* =============================================================================
* estimate_ovb.do — Figure 2, Figure A3, Tables A4 and A5
*
* Omitted variable bias (OVB) analysis. For each outcome:
*   1. Regress outcome on design covariates absorbing teacher FE to extract
*      the teacher effect (teach_fe = xbd - xb from areg).
*   2. Regress outcome on design + excluded covariates + twin FE (via reghdfe),
*      then form the fitted value of the excluded covariates + twin FE
*      (covadj_predict). This is the "omitted variables fit."
*   3. Regress covadj_predict on teach_fe: the slope measures the degree to
*      which teacher effects are correlated with omitted variables.
*   4. Binscatter plots of covadj_predict vs. teach_fe (Figures 2 and A3).
*
* Table A4: OVB tests for short-run outcomes (test scores, behaviors, study)
* Table A5: OVB tests for CJC long-run outcomes (any arrest, criminal, index,
*           incarceration)
* Figures A3c-d: OVB for academic long-run (college-bound, GPA)
* =============================================================================

*** 0) Load and prep data and set options
clear all
clear matrix
set more off

* Load data and options
do code/set_options.do
do code/preamble.do

* The teacher effect extracted from areg is the RHS variable in the OVB test
global regvar "teach_fe"

*** 1) OVB tests for short-run outcomes (Table A4, Figures 2a-b)
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

*** 2) OVB tests for long-run academic outcomes (Figures A3c-d)
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

*** 3) OVB tests for CJC long-run outcomes (Table A5, Figures A3a-b)
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

