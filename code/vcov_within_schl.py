#teacher ustat
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


#####################################
### 0) Options/globals
##################################### 
# Options
ns = 500

# Color options
col_grad = 'blue'
col_gpa = 'royalblue'
col_college = 'purple'

col_aoc_traff = 'gold'
col_crimeany = 'orange'
col_crime = 'peru'
col_aoc_index = 'red'
col_aoc_incar = 'darkred'

### 0) Helper functions
def withinOnly_school(sX, sY, sids, yearWeighted = False):
    sdevs = []
    sdevs_ses = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    for id in tqdm(unique_ids):
        left = sX.copy()
        left[sids != id] = np.nan
        right = sY.copy()
        right[sids != id] = np.nan

        # Only use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
            if yearWeighted:
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
            else:
                nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
            try:
                sdevs_ses += [ustat.vcv_samp_var(left, right)*nteach**2]
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
            except:     # If not enough obs to compute SE, skip it
                pass

    within = np.nansum(sdevs) / np.nansum(totler)
    within_se = np.nansum(sdevs_ses) / np.nansum(totler)**2
    return within, within_se

def tabfunc(effects, outcome, sids):
    results = pd.DataFrame(columns=list(effects.keys()) + list(outcome.keys()), index=effects.keys())
    results_vcv = pd.DataFrame(columns=list(effects.keys()) + list(outcome.keys()), index=effects.keys())
    for row in results.index:
        for col in effects.keys():
            print(row,col)
            within, se = withinOnly_school(effects[row], effects[col], sids)
            results.loc[row, col] = within
            results_vcv.loc[row, col] = se
        for col in outcome.keys():
            within, se = withinOnly_school(effects[row], outcome[col], sids)
            results.loc[row, col] = within
            results_vcv.loc[row, col] = se

    XX = results.iloc[:,:len(effects)].values.astype(float)
    xY = results.iloc[:,len(effects):].values.astype(float)
    beta = np.linalg.inv(XX).dot(xY)

    ### Parametric bootstrap for SES 
    mu = results.values.astype(float).ravel()
    var = np.diag(results_vcv.values.astype(float).ravel())
    combined = {**effects,**outcome} 

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
        within, se = withinOnly_school(outcome[col], outcome[col], sids)        
        results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(
                within**0.5, se**0.5)
        results.loc['$R^2$', col] = "{:4.3f}".format(
                    beta[:,c].dot(XX).dot(beta[:,c])/within)
    return results


### Within school with constant effects
tresids = pd.read_stata("temp/teach_mean_resids.dta").drop('teachid', axis=1)

cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]
gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

sids = tresids.filter(regex='^school_fe', axis=1).values[:,:]

### Covariance with short-run outcomes
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study}
outcome = {'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'Graduation':grad , 'College attendance':college}
results = tabfunc(effects,outcome,sids)

results.to_latex('tables/tableA9.tex', 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 


