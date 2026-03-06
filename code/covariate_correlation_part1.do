* =============================================================================
* covariate_correlation_part1.do — In-text calculations (Stata step)
*
* Computes the correlation between teacher effects and the design covariates,
* used to quantify how much of the variation in teacher effects could be
* explained by observable covariates. The result is an upper bound on the
* bias from sorting on observables.
*
* Procedure:
*   1. Regress test scores on design covariates absorbing teacher FE
*   2. Save the coefficient vector (beta) and variance-covariance matrix (sigma)
*   3. Manually reconstruct the design matrix X (expanding factor variables
*      and interactions) so that teacher-year means of X can be paired with
*      sigma in the Python step
*   4. Collapse X to teacher-year means and save to temp/teach_mean_covars.dta
*   5. Save sigma to temp/sigma.dta
*
* The Python step (covariate_correlation_part2.py) then computes:
*   E_j[x_j' Sigma x_j'] across all teacher cross-year pairs, which gives
*   the average bias in the variance estimate due to covariate correlations.
*
* Output:
*   temp/teach_mean_covars.dta — teacher-year mean covariate values
*   temp/sigma.dta — variance-covariance matrix of regression coefficients
* =============================================================================
clear all
clear matrix
set more off

* Load data and options
do code/set_options.do
do code/preamble.do

* Regress test scores on design covariates absorbing teacher FE
areg testscores $covdesign, abs(teachid) robust
assert e(sample) == 1    // All data should be used for test score VA

* Save the coefficient vector and VCV matrix from the regression
matrix beta = e(b)
matrix sigma = e(V)

* Save fitted values to verify the manual design matrix construction
predict xb, xb

* Manually reconstruct the design matrix X by parsing Stata's coefficient
* names (which encode factor variables and interactions like "2.grade#1.subj").
* This is necessary because Stata does not export the design matrix directly.
local beta_names : colnames beta
local vcount = 1
foreach col of local beta_names {
    gen _x`vcount' = 1
    tokenize "`col'", parse("#")
    local i = 1
    di ""
    di "`col'"
    while "``i''" != "" {
        if "``i''" == "#" {
        }
        else {
            tokenize "``i''", parse(".")
            if "`3'" != "" {
                if ("`1'" != "c") | ("`1'" != "co") {
                    if ("`1'" == "o") {
                        di "multiplying by `3'"
                        qui replace _x`vcount' = _x`vcount'*`3'
                    }
                    else {
                        local ind : subinstr local 1 "b" "", all
                        local ind : subinstr local ind "o" "", all
                        local ind : subinstr local ind "c" "", all
                        if ("`1'" == "c") | ("`1'" == "co") {
                            di "multiplying by `3'"
                            qui replace _x`vcount' = _x`vcount'*`3'                            
                        }
                        else {
                            di "multiplying by (`3' == `ind')"
                            qui replace _x`vcount' = _x`vcount'*(`3' == `ind')
                        }
                    }
                }
                else  {
                    di "multiplying by `3'"
                    qui replace _x`vcount' = _x`vcount'*`3'
                }
            }
            else {
                di "replacing with `col'"
                replace _x`vcount' = `col'
            }
            tokenize "`col'", parse("#")
        }
        local i = `i' + 1        
    }

    local ++ vcount
}
di "`vcount'"


* Check that it is computed correctly
gen touse_test = runiform() < 10000/_N
gen idvar = _n
putmata idvar if touse_test
mkmat _x* if touse_test, matrix(X)
mata: X = st_matrix("X")
mata: beta = st_matrix("beta")
mata: yhat = X*beta'
getmata (xb_test) = yhat, id(idvar)
gen xb_dif = abs(xb_test - xb)
su xb_dif, d
assert r(max) < .0001

* Save as extra variabls
keep teachid year _x*

* Collapse by teacher year
collapse (mean) _x*, by(teachid year)

sort teachid year
by teachid: gen obs = _n

* Save for use in python
save temp/teach_mean_covars.dta, replace

* Save VCV for the gamma
clear
svmat sigma
save temp/sigma.dta, replace

