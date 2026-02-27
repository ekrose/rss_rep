**************************************************
* Make table
**************************************************
clear all
clear matrix
set more off

*** (1) Build matrices of SD of teacher effects and their standard errors
* Load variance (and SE) estimates that have been calculated in Python
preserve
use "temp/hetero_SDs.dta", clear

levelsof subgroup, local(_subgroups)
local num_groups: list sizeof local(_subgroups)

levelsof outcome, local(_outcomes)
local num_outcomes: list sizeof local(_outcomes)


mat tab_est = J(`num_outcomes',`num_groups',.) 
mat tab_se = J(`num_outcomes',`num_groups',.) 

local row_counter = 1
foreach y of local _outcomes{
    local col_counter = 1
    foreach g of local _subgroups{
        qui sum value if statistic == "SD" & outcome == "`y'" & subgroup == "`g'"
        mat tab_est[`row_counter', `col_counter'] = r(mean)

        qui sum value if statistic == "SDse" & outcome == "`y'" & subgroup == "`g'"
        mat tab_se[`row_counter', `col_counter'] = r(mean)

        local ++col_counter
    }
    local ++row_counter
}

dis "Finished building matrices of SD of teacher effects and their standard errors"
mat li tab_est
mat li tab_se
restore

*** (2) export to latex uisng STATA

** build templete
* Define variables as the number of outcomes (i.e., coefficients/rows)
capture drop _*
clear
set obs 100
gen _zz = runiform() 

local name_outcomes ""
local row_counter = 1
foreach y of local _outcomes{
    gen _yy`row_counter' = runiform()
    label var _yy`row_counter' "`y'"
    local name_outcomes "`name_outcomes' `y'"
    local ++row_counter
}
reg _zz _yy*, noconst

esttab, se
mat list r(coefs)
matrix c = r(coefs)

local numrows `=rowsof(c)'
local rnames : rownames c
assert `numrows' == `num_outcomes'


eststo clear
local col_counter = 1
foreach yy of local _subgroups {
    matrix b = tab_est[1..., `col_counter']'
    matrix colnames b = `rnames'

    matrix se = tab_se[1..., `col_counter']'
    matrix colnames se = `rnames'   

    ereturn post b
    estadd matrix se
    eststo

    local ++col_counter
}


esttab using tables/tableA13.tex, tex se label nostar replace mtitles("White" "Non-White" "Boys" "Girls" "Yes" "No" "High" "Low" ) mgroups("Race" "Sex" "Econ. disadvantaged" "Arrest risk", pattern(1 0 1 0 1 0 1 0 ) prefix(\multicolumn{@span}{c}{) suffix(}) span erepeat(\cmidrule(lr){@span})) 



