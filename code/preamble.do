* Create any variables as needed
encode subject, gen(subj)

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

* Add first PCA measures
foreach var in readscal mathscal {
	assert missing(`var') == 0
}
foreach prefix in "" "lead1_" "lead2_" "lead3_" "lead4_" {
	qui pca `prefix'mathscal `prefix'readscal
	qui predict `prefix'cogpca, score
	
	qui pca `prefix'homework `prefix'freeread `prefix'watchtv
	qui predict `prefix'studypca, score

	* Generate subject specific test scores
	gen `prefix'testscores = `prefix'cogpca if subject == "hr"
	replace `prefix'testscores = `prefix'mathscal if subject == "math"
	replace `prefix'testscores = `prefix'readscal if subject == "eng"

}

foreach var of varlist lead1_daysabs lead1_any_discp lead_grade_rep {
	capture drop _tmp
	bys year grade: egen _tmp = sd(`var')
    gen `var'_norm = `var' / _tmp
}
qui pca lead1_daysabs_norm lead1_any_discp_norm lead_grade_rep_norm
qui predict behavpca, score
replace behavpca = -behavpca

capture drop _tmp
bys year grade: egen _tmp = sd(behavpca)
gen behavpca_normalized = behavpca / _tmp
drop _tmp

foreach var of varlist lead2_daysabs lead2_any_discp lead1_grade_rep {
	capture drop _tmp
	bys year grade: egen _tmp = sd(`var')
    gen `var'_norm = `var' / _tmp
}
qui pca lead2_daysabs_norm lead2_any_discp_norm lead1_grade_rep_norm
qui predict lead1_behavpca, score
replace lead1_behavpca = -lead1_behavpca
replace lead1_behavpca = . if grade == 8

foreach var of varlist lead3_daysabs lead3_any_discp lead2_grade_rep {
	capture drop _tmp
	bys year grade: egen _tmp = sd(`var')
    gen `var'_norm = `var' / _tmp
}
qui pca lead3_daysabs_norm lead3_any_discp_norm lead2_grade_rep_norm
qui predict lead2_behavpca, score
replace lead2_behavpca = -lead2_behavpca

foreach var of varlist lead4_daysabs lead4_any_discp lead3_grade_rep {
	capture drop _tmp
	bys year grade: egen _tmp = sd(`var')
    gen `var'_norm = `var' / _tmp
}
qui pca lead4_daysabs_norm lead4_any_discp_norm lead3_grade_rep_norm
qui predict lead3_behavpca, score
replace lead3_behavpca = -lead3_behavpca

* Drop any means of excluded variables
foreach var of global covsadj {
	drop *_mean_`var'*
}

* Dummy out any missing controls in design variables and adjust variables
foreach var of global covsadj {
	if strpos("`var'", "#") == 0 {
		qui desc `var', varlist
		local _tmp = r(varlist)
		foreach v of local _tmp {
			gen madj_`v' = missing(`v')
			replace `v' = 0 if missing(`v')
		}
	}
}
global covsadj = "${covsadj} madj_*"

foreach var of global covdesign {
	if strpos("`var'", "#") == 0 {
		qui desc `var', varlist
		local _tmp = r(varlist)
		foreach v of local _tmp {
			gen mdes_`v' = missing(`v')
			replace `v' = 0 if missing(`v')
		}
	}
}

* Fix missing values of means
foreach v in  exc_not exc_aig exc_behav exc_educ{
	** Replace missings
	local vv = "s_mean_`v'"
	gen `vv'_mdes = missing(`vv')
	replace `vv' = 0 if missing(`vv')

	** Replace missings in means
	local vv = "sgyts_mean_`v'"
	gen `vv'_mdes = missing(`vv')
	replace `vv' = 0 if missing(`vv')
}

* One more
gen sgyts_mean_disadv_mdes = missing(sgyts_mean_disadv)
replace sgyts_mean_disadv = 0 if missing(sgyts_mean_disadv)

global covdesign = "${covdesign} mdes_* *_mdes"


* Predict cognitive and crime outcomes based on parent chars and lag (Missing dummies)
foreach ll in "${outcome}" aoc_any behavpca studypca {
	qui reg `ll' ${covsadj} 
	capture predict all_`ll'_idx, xb	

	qui reg `ll' pared_* madj_pared_*
	capture predict par_`ll'_idx, xb	

	qui reg `ll' lag2_mathscal lag2_readscal madj_lag2_mathscal madj_lag2_readscal
	capture predict lag_`ll'_idx, xb		
} 

* VAM calculation
capture program drop fun_vam
program define fun_vam
	args outcome cov_opt
		* args:
		* 	"outcome": outcome variable of interest (e.g., anysc1621, mathscal) 
		*	"cov_opt": indicator for whether or not to include the "excluded" adjustment controls in the VAM calculation. 

	if ("${vam_measure}" == "jackson") | ("${vam_measure}" == "naive") | ("${vam_measure}" == "blp") | ("${vam_measure}" == "none") {

		*** 1) Calculate residuals and LOO VAMs (w/t EB shrinkage)
		if `cov_opt' == 1 {
			qui areg `outcome' ${covdesign} ${covsadj}, abs(teachid) 	
		}
		else {
			qui areg `outcome' ${covdesign}, abs(teachid)	
		}
		predict score_r, dresiduals
		predict xhat, xb
		predict xhatd, xbd
		gen teach_fe = xhatd - xhat

		*** 2) Build teacher-year-subject level dataset
		bysort teachid year: egen double ebar_jt = mean(score_r)
		qui sum ebar_jt
		replace ebar_jt = ebar_jt - r(mean)

		* Recentered leave-year-out ebar
		bysort teachid: egen double res_j_total = total(score_r)
		bysort teachid year: egen double res_jt_total = total(score_r)
		bysort teachid: egen double n_j = count(score_r)
		bysort teachid year: egen double n_jt = count(score_r)

		gen double loo_ebar_jt = (res_j_total - res_jt_total)/(n_j - n_jt)
		qui sum loo_ebar_jt
		replace loo_ebar_jt = loo_ebar_jt - r(mean)
		bysort teachid: egen nyears_j = nvals(year)

		* Leave out school-grade
		bysort teachid school_fe grade: egen double res_j_total_schgrade = total(score_r)
		bysort teachid school_fe grade: egen double n_j_schgrade = count(score_r)
		gen loo_ebar_schgrade = (res_j_total - res_j_total_schgrade) / (n_j - n_j_schgrade)
		gen ebar_schgrade = res_j_total_schgrade / n_j_schgrade

		* Leave out school
		bysort teachid school_fe: egen double res_j_total_school = total(score_r)
		bysort teachid school_fe: egen double n_j_school = count(score_r)
		gen loo_ebar_school = (res_j_total - res_j_total_school) / (n_j - n_j_school)
		gen ebar_school = res_j_total_school / n_j_school
		
		* Get total students and scores from other years
		preserve 
		collapse (mean) n_jt res_jt_total, by(teachid year subject)
		rename n_jt n_jtm1 
		rename res_jt_total res_jtm1_total
		rename year merge_year
		tempfile twoyearlom
		save `twoyearlom', replace
		restore

		* Leave outs of additional years
		gen double loo_ebar_jt2 = (res_j_total - res_jt_total)
		gen double loo_denom_tmp = (n_j - n_jt)

		foreach k of numlist 1/2 {
			capture drop merge_year
			gen merge_year = year - `k'
			merge m:1 teachid merge_year subject using `twoyearlom', nogen keep(1 3)
				replace loo_ebar_jt2 = loo_ebar_jt2 - res_jtm1_total
				replace loo_denom_tmp = loo_denom_tmp - n_jtm1
				if `k' == 1 {
					gen loo_ebar_jtm1 = loo_ebar_jt2/loo_denom_tmp
				}
				drop res_jtm1_total n_jtm1

			capture drop merge_year
			gen merge_year = year + `k'
			merge m:1 teachid merge_year subject using `twoyearlom', nogen keep(1 3)
				replace loo_ebar_jt2 = loo_ebar_jt2 - res_jtm1_total
				replace loo_denom_tmp = loo_denom_tmp - n_jtm1
				if `k' == 1 {
					gen loo_ebar_jtm1p1 = loo_ebar_jt2/loo_denom_tmp
					gen loo_ebar_jtp1 = (res_j_total - res_jt_total - res_jtm1_total)/(n_j - n_jt - n_jtm1)
				}
				drop res_jtm1_total n_jtm1
		}
		
		replace loo_ebar_jt2 = loo_ebar_jt2/loo_denom_tmp
		replace loo_ebar_jt2 = . if loo_denom_tmp <= 0 | loo_denom_tmp == .
		drop loo_denom_tmp

		qui sum loo_ebar_jt2
		replace loo_ebar_jt2 = loo_ebar_jt2 - r(mean)

		qui sum loo_ebar_jtm1
		replace loo_ebar_jtm1 = loo_ebar_jtm1 - r(mean)

		if "$vam_measure" == "none" {
			gen vam  = loo_ebar_jt
			gen vaml2 = loo_ebar_jtm1
			gen vaml3 = loo_ebar_jtm1p1
			gen vaml4 = loo_ebar_jt2
			gen vamlf1 = loo_ebar_jtp1
			gen resid = score_r - vam
			gen yhat = xhat + vam
		}
		else if "$vam_measure" == "blp" {
			* BLP using LOO mean
			reg ebar_jt ibn.nyears_j#c.loo_ebar_jt
			predict vam, xb
			gen resid = score_r - vam
			gen yhat = xhat + vam

			* BLP using leave out t and t-1
			reg ebar_jt ibn.nyears_j#c.loo_ebar_jtm1
			predict vaml2, xb

			* BLP using leave out t, t-1 and t+1
			reg ebar_jt ibn.nyears_j#c.loo_ebar_jtm1p1
			predict vaml3, xb

			* BLP using leave out t, t-1, t-2, t+1 and t+2
			reg ebar_jt ibn.nyears_j#c.loo_ebar_jt2
			predict vaml4, xb
		}

		else if "$vam_measure" == "naive" {

			*** lambda (shrinkage)
			* _tmp_n_jt
			capture drop _tmp_n_jt
			bysort teachid year: gen double _tmp_n_jt = n_jt^2 if _n == 1
			
			capture drop total_n_jt2_loo
			bysort teachid: egen total_n_jt2_loo = total(_tmp_n_jt)
			replace total_n_jt2_loo = total_n_jt2_loo - n_jt^2
			capture drop _tmp_n_jt

			capture drop _lambda_jt
			gen double _lambda_jt = (total_n_jt2_loo/(n_jt_other^2))*`var_classroom' + `var_e'/(n_jt_other) + `var_teacher_effects'
			bysort teachid year: replace _lambda_jt = . if _n > 1
			qui sum _lambda_jt
			capture drop lambda
			gen double lambda = `var_teacher_effects'/r(mean)

			* VAM with naive shrinkage
			capture drop vam_naiveeb
			gen double vam_naiveeb = lambda*loo_ebar_jt
		}
		else if "${vam_measure}" == "jackson"{
			*** 3) Notes: To conduct Jackson/Kane and Staigr EB procedure we need several variances
			* Variance of residuals (=Var_e + Var_teachers + Var_classroom) 
			qui sum score_r
			local var_res=r(sd)^2

			* Variance of within-class error term (Var_e)
			capture drop classYear
			egen classYear = group(teachid year)

			areg score_r, abs(classYear) 
			capture drop _tmpres
			predict _tmpres, residuals
			qui sum _tmpres
			local var_e=r(sd)^2
			drop _tmpres


			* Variance of teacher effects --- will be done by random matches of classrooms
			* We will use both methods and compare. But first built data in a format of a classroom residuals
			preserve
			keep teachid year n_jt ebar_jt
			duplicates drop 

			* Variance of teacher effects using random matching of classrooms
			* random match of classrooms within a teacher
			global numiters = 200
			mat sims = J(${numiters},1,.)
			local cc = 1
			forvalues itr = 1(1)$numiters{
				dis "Working on iteration `itr'"
				capture drop _draw _random_match
				gen double _draw = runiform() 
				sort teachid _draw
				by teachid: gen _random_match = _n

				forvalues ss = 1(1)2{
					capture drop *ebar`ss' 
					gen _ebar`ss' = ebar_jt if _random_match == `ss'	
					by teachid: egen ebar`ss' = max(_ebar`ss')
				}
				corr ebar1 ebar2 if _random_match == 1, covariance
				mat sims[`cc',1] = r(cov_12)
				local ++cc
			}
			svmat sims
			rename sims1 sims
			sum sims, d
			local var_teacher_effects = r(p50)
			restore

			* Variance of classroom effects
			local var_classroom = `var_res' - `var_e' - `var_teacher_effects'

			*** 4) EB shrinkage
			local cc = 1
			levelsof year, local(_listyears)
			foreach yr of local _listyears{
				dis "Working on year: `yr'"
				preserve
				keep teachid year n_jt ebar_jt
				duplicates drop 
				drop if year == `yr'

				gen double h_jt = 1/(`var_classroom' + `var_e'/n_jt )
				gen h_by_ebar_jt=ebar_jt/(`var_classroom' + `var_e'/n_jt )
				bysort teachid: egen denominator=total(h_jt)
				bysort teachid: egen numerator=total(h_by_ebar_jt)
				gen double ebar_j= numerator/denominator
				gen double vam_eb = ebar_j*(`var_teacher_effects'/(`var_teacher_effects'+(1/denominator)))

				* Save to a temp file
				keep vam_eb teachid
				duplicates drop
				gen year = `yr'

				if `cc' == 1{
					tempfile vamEB
					save vamEB, replace				
				}
				else{
					append using vamEB
					save vamEB, replace
				}
				restore
				local ++cc
			}

			* Merge back
			merge m:1 teachid year using vamEB, nogen
		}

		dis "SD of teacher effects is: `var_teacher_effects'"
		dis "SD of classroom shocks is: `var_classroom'"
		dis "SD of individual within-classroom residuals is: `var_e'"
	}
	else if "${vam_measure}" == "chetty"{
		*Chetty's version of vam (he just uses something he calls "score")
		*Note that class for us doesn't vary by subject (as of 8/5/20), so there is some sense of redundancy in these definitions
		capture drop classChetty score_r
		egen classChetty = group(teachid year)
		global dtadir "temp/"
		if `cov_opt' == 1{
			vam `outcome', teacher(teachid) year(year) class(classChetty) controls(${covdesign} ${covsadj}) output("$dtadir/tfx") tfx_resid(teachid) driftlimit(5) data(merge tv score_r) quasiexperiment
		}
		else{
			vam `outcome', teacher(teachid) year(year) class(classChetty) controls(${covdesign}) output("$dtadir/tfx") tfx_resid(teachid) driftlimit(5) data(merge tv score_r) quasiexperiment
		}		
		rename tv vam
		rename tv_2yr_l vaml2
		rename tv_ss vaml4
		gen vaml3 = vaml4
		gen yhat = vaml3
	}
	else if "${vam_measure}" == "MElogit"{
		
		melogit `outcome' $covdesign || teachid:
		* estimates store melgtNoDemo

		capture drop vam_melgt
		gen vam_melgt = .
		levelsof year, local(_tmp)
		foreach yy of local _tmp{
			dis `Working on LOO VAM/EBmeans of year: yy'
			capture drop _tmpyear*
			predict _tmpyear if year != `yy', remeans
			bysort teachid: egen double _tmpyear2 = max(_tmpyear)
			assert vam_melgt ==. if year == `yy'
			replace vam_melgt =  _tmpyear2 if year == `yy'
		}
	}
	else {
		dis "NO valid VAM measure specified"
	}
end

