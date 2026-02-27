import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys

# For parallel processing
from multiprocessing import Pool
from functools import partial

import funcs_vcov_ustats as ustat

# Graphing
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
plt.style.use('ggplot')

# Seed
np.random.seed(671238)

# Color options
col_grad = 'blue'
col_gpa = 'royalblue'
col_college = 'purple'

col_aoc_traff = 'gold'
col_crimeany = 'orange'
col_crime = 'peru'
col_aoc_index = 'red'
col_aoc_incar = 'darkred'

### 0) Function for table making
def tabfunc(effects, outcome):
    results = pd.DataFrame(columns=list(effects.keys()) + list(outcome.keys()), index=effects.keys())
    results_vcv = pd.DataFrame(columns=list(effects.keys()) + list(outcome.keys()), index=effects.keys())
    for row in results.index:
        for col in effects.keys():
            results.loc[row, col] = ustat.varcovar(effects[row], effects[col])
            results_vcv.loc[row, col] = ustat.vcv_samp_covar(effects[row], effects[col])
        for col in outcome.keys():
            results.loc[row, col] = ustat.varcovar(effects[row], outcome[col])
            results_vcv.loc[row, col] = ustat.vcv_samp_covar(effects[row], outcome[col])

    XX = results.iloc[:,:len(effects)].values.astype(float)
    xY = results.iloc[:,len(effects):].values.astype(float)
    beta = np.linalg.inv(XX).dot(xY)

    ### Parametric bootstrap for SES 
    mu = results.values.astype(float).ravel()
    var = np.diag(results_vcv.values.astype(float).ravel())
    combined = {**effects,**outcome}
    
    # Fill in rest of VCV matrix
    row = 0
    for row1 in results.index:
        for col1 in list(effects.keys()) + list(outcome.keys()):
            col = 0
            for row2 in results.index:
                for col2 in list(effects.keys()) + list(outcome.keys()):
                    print("Working on ({},{}) ({},{})".format(row1,col1,row2,col2))
                    if col > row:
                        var[row,col] = ustat.ustat_samp_covar(
                                    combined[row1], combined[col1], combined[row2], combined[col2])
                    col+=1 
            row+=1 
    var = var + np.triu(var,1).T     

    ns = 500
    bs_ests = np.zeros(shape=(beta.shape[0],beta.shape[1],ns))
    for n in range(ns):
        newres = np.random.multivariate_normal(mu,var).reshape(results.shape)
        bXX = newres[:,:len(effects)]
        bxY = newres[:,len(effects):]
        bs_ests[:,:,n] += np.linalg.inv(bXX).dot(bxY)
    sds = np.power(np.var(bs_ests,2),0.5)

    # make table
    results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
    for r, row in enumerate(results.index):
        for c, col in enumerate(results.columns):
            results.loc[row, col] = "{:4.3f} ({:4.3f})".format(beta[r, c], sds[r, c])
    for c, col in enumerate(results.columns):
        results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(
                ustat.varcovar(outcome[col], outcome[col])**0.5,
                    ustat.sd_samp_var(outcome[col])**0.5)
        results.loc['$R^2$', col] = "{:4.3f}".format(
                    beta[:,c].dot(XX).dot(beta[:,c])/ustat.varcovar(outcome[col], outcome[col]))
    return results

#######################################################################
### 1) Multivariate regression of short-run VAM on long-run outcomes
#######################################################################

### Option 1: Using leave-one-out teacher-year pairs
tresids = pd.read_stata(
        "temp/teach_mean_resids.dta".format(
        spec, droppval)).drop('teachid', axis=1)

cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]
crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

# Regression decomposition
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study}
outcome = {'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'Graduation':grad , 'College attendance':college}

### Multivariate infeasable regression
results = tabfunc(effects,outcome)

results.to_latex(
        'tables/table4.tex'.format(spec,droppval), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 


### Option 2: Using leave-one-out teacher-school pairs
tresids = pd.read_stata(
    "temp/teachSchl_mean_resids.dta".format(
            spec, droppval)).drop('teachid', axis=1)

cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]
crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

# Regression decomposition
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study}
outcome = {'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'Graduation':grad , 'College attendance':college}

### Multivariate infeasable regression
results = tabfunc(effects,outcome)

results.to_latex(
        'tables/tableA8.tex'.format(spec,droppval), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 











