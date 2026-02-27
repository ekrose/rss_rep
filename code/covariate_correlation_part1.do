clear all
clear matrix
set more off

* Load data and options 
do code/set_options.do
do code/preamble.do

* Get the VCV for test scores
areg testscores $covdesign, abs(teachid) robust
assert e(sample) == 1    // All data is used for test score VA

* Save VCV and beta
matrix beta = e(b)
matrix sigma = e(V)

* Save fits to check procedure
predict xb, xb 

* Create design matrix manually
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

