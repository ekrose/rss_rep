#teacher ustat
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys

# For parallel processing
from multiprocessing import Pool
from functools import partial

os.chdir('/accounts/projects/crwalters/cncrime/teachers_final/code/')
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
spec = 7
droppval = 0
ns = 300
ncpus = 16
# ns = 3
# ncpus = 4
# ncpus = np.nan

np.random.seed(93293483)
gamma_draw_mat = np.random.exponential(scale=1, size = (55000,17,ns))


# Color options
col_grad = 'blue'
col_gpa = 'royalblue'
col_college = 'purple'

col_aoc_traff = 'gold'
col_crimeany = 'orange'
col_crime = 'peru'
col_aoc_index = 'red'
col_aoc_incar = 'darkred'

#######################################################################
### 0) Short-run outcomes variance-covariance 
#######################################################################

if 0:
    # Load the data
    tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_hetero_samp100.dta".format(
            spec, droppval)).drop('teachid', axis=1)
    cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
    # math = tresids.filter(regex='^math_', axis=1).values[:,:]
    # eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
    study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
    behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
    # oss = tresids.filter(regex='^lead1_oss_r', axis=1).values[:,:]

    # effects = {'Test scores':cog, 'Math scores':math, 'Reading scores':eng, 'Study skills':study, 'Behaviors':behave}
    effects = {'Test scores':cog, 'Study skills':study, 'Behaviors':behave}
    results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

    ## Main var covar
    # First populate SDs
    for idx, row in enumerate(results.columns):
        sdev = ustat.sd_func(effects[row], effects[row])
        _gamma = gamma_draw_mat[:effects[row].shape[0], :effects[row].shape[1],:].copy()
        se = ustat.bsci_varcovar(effects[row], effects[row], ustat.BB_sd_func, gamma_mat = _gamma, ntasks = ncpus)
        results.loc[row, row] = "{:4.3f} ({:4.3f})".format(sdev, se)

    # Now add correlations
    for idx, row in enumerate(results.columns):
        for col in results.iloc[idx+1:].index:
            sdev = ustat.correl_func(effects[row], effects[col])
            _gamma = gamma_draw_mat[:effects[row].shape[0], :effects[row].shape[1],:].copy()
            se = ustat.bsci_varcovar(effects[row], effects[col], ustat.BB_correl_func, gamma_mat = _gamma, ntasks = ncpus)
            results.loc[row, col] = "{:4.3f} ({:4.3f})".format(sdev, se)

    # Save to latex
    results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/varcovar_TeacherSchlHetero_spec{}_droppval{}_bt{}_shortrun_samp100.tex'.format(spec,droppval, ns), 
        na_rep='', escape=False, 
        multicolumn_format = 'c',
        column_format = 'c' * int(results.shape[1] + 1)
        )  

#######################################################################
### 1) Implied regression: HETERO SCHOOL EFFECTS
#######################################################################

def withinOnly_school(sX,sY, sids, yearWeighted = False):
    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    expecs_left = []
    expecs_right = []
    for id in tqdm(unique_ids):
        left = sX.copy()
        left[sids != id] = np.NaN
        right = sY.copy()
        right[sids != id] = np.NaN

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
            if yearWeighted:
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
            else:
                nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
            sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
            totler += [nteach]
            expecs_left += [np.nanmean(left)]
            expecs_right += [np.nanmean(right)]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within 

def correl_func_school(sX,sY, sids, yearWeighted = False): 
    return withinOnly_school(sX,sY, sids, yearWeighted = yearWeighted)/np.power(withinOnly_school(sX,sX, sids, yearWeighted = yearWeighted)*withinOnly_school(sY,sY, sids, yearWeighted = yearWeighted), 0.5) 


def sd_func_school(sX,sY, sids, yearWeighted = False): 
    return np.power(withinOnly_school(sX,sY, sids, yearWeighted = yearWeighted), 0.5)


tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_hetero_samp100.dta".format(spec, droppval))

cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]
gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

sids = tresids.filter(regex='^sid', axis=1).values[:,:]

### Covariance with short-run outcomes
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study}
outcome = {'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'Graduation':grad , 'College attendance':college}
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())

### Multivariate infeasable regression
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
for row in results.index:
    for col in results.columns:
        results.loc[row, col] = withinOnly_school(effects[row], effects[col], sids)
XX = results.values.astype('float')
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
for row in results.index:
    for col in results.columns:
        results.loc[row, col] = withinOnly_school(effects[row], outcome[col], sids)
xY = results.values.astype('float')
beta = np.linalg.inv(XX).dot(xY)
vary = [withinOnly_school(outcome[col], outcome[col], sids) for col in outcome.keys()]
vary_se = [ustat.bsci_varcovar(outcome[col], outcome[col], func = ustat.BB_sd_func, gamma_mat = gamma_draw_mat[:outcome[col].shape[0], :outcome[col].shape[1],:].copy(), ntasks = ncpus) for col in outcome.keys()]
varx = [beta[:,k].T.dot(XX).dot(beta[:,k]) for k in range(beta.shape[1])]


# Now get SEs using BB
def BB_withinOnly_school(sX,sY, sids, gamma_draw, yearWeighted = True):
    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    expecs_left = []
    expecs_right = []
    for id in tqdm(unique_ids):
        left = sX.copy()
        left[sids != id] = np.NaN
        right = sY.copy()
        right[sids != id] = np.NaN

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
            if yearWeighted:
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
            else:
                nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
            sdevs += [ustat.BB_varcovar(left, right, gamma_draw, yearWeighted = yearWeighted)*nteach]
            totler += [nteach]
            expecs_left += [np.nanmean(left)]
            expecs_right += [np.nanmean(right)]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within 

def BB_multivar(effects, sids, gamma_draw, yearWeighted=True):
    results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
    for row in results.index:
        for col in results.columns:
            results.loc[row, col] = BB_withinOnly_school(effects[row], effects[col], sids, gamma_draw[:effects[row].shape[0], :effects[row].shape[1]], yearWeighted = yearWeighted)
    XX = results.values.astype('float')
    results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
    for row in results.index:
        for col in results.columns:
            results.loc[row, col] = BB_withinOnly_school(effects[row], outcome[col], sids, gamma_draw[:effects[row].shape[0], :effects[row].shape[1]], yearWeighted = yearWeighted)
    xY = results.values.astype('float')
    beta = np.linalg.inv(XX).dot(xY)
    return beta

def mvar_bs_helper(l, func, gamma_mat, effects, sids):
        return func(effects, sids, gamma_mat[:,:,l])

def mvar_bsci_varcovar(effects, sids, func, gamma_mat, ntasks = np.nan):
    nsims = gamma_mat.shape[2]
    
    if np.isnan(ntasks):
        result = []
        for l in range(nsims):
            result += [func(effects, sids, gamma_mat[:,:,l]),]    
    else:
        _foo = partial(mvar_bs_helper, func=func, effects = effects, sids=sids,  gamma_mat = gamma_mat)
        with Pool(ntasks) as p:
            result = list(tqdm(p.imap(_foo, range(nsims)), total=nsims))
    result = np.array(result, dtype=np.float64)
    _se = np.nanstd(result, axis=0)
    return _se

_gamma = gamma_draw_mat[:effects['Test scores'].shape[0], :effects['Test scores'].shape[1],:].copy()
beta_se = mvar_bsci_varcovar(effects, sids, func = BB_multivar, gamma_mat = _gamma, ntasks = ncpus)


# make table
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
for r, row in enumerate(results.index):
    for c, col in enumerate(results.columns):
        results.loc[row, col] = "{:4.3f} ({:4.3f})".format(beta[r, c], beta_se[r, c])
        # results.loc[row, col] = "{:4.3f}".format(beta[r, c])
for c, col in enumerate(results.columns):
    results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(vary[c]**0.5, vary_se[c])
    # results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f}".format(vary[c]**0.5)
    results.loc['$R^2$', col] = "{:4.3f}".format(varx[c]/vary[c])

results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlHetero_spec{}_droppval{}_bt{}_longrun_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 

results.loc[:, ['12th grade GPA', 'Graduation', 'College attendance']].to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlHetero_spec{}_droppval{}_bt{}_longrun_academic_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 


results.loc[:, ['Any CJC', 'Criminal arrest', 'Index crime', 'Incarceration']].to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlHetero_spec{}_droppval{}_bt{}_longrun_cjc_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 


#######################################################################
### 2) Implied regression: CONSTANT TEACHER EFFECTS
#######################################################################

def within_school_constantEffect(sX,sY, sids, yearWeighted = True):

    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    sids_ls = []
    for id in tqdm(unique_ids):
        left = sX[(sids == id).max(axis=1)]
        right = sY[(sids == id).max(axis=1)]

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):

            # Only teachers with some non-missing observations in school "id" for outcomes "X" or "Y"
            if ((~np.isnan(sY[(sids == id)])).sum() > 0) | ((~np.isnan(sY[(sids == id)])).sum() > 0):
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                sids_ls += [id]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within

def correl_func_school(sX,sY, sids, yearWeighted = True): 
    return withinOnly_school(sX,sY, sids, yearWeighted = yearWeighted)/np.power(within_school_constantEffect(sX,sX, sids, yearWeighted = yearWeighted)*within_school_constantEffect(sY,sY, sids, yearWeighted = yearWeighted), 0.5) 


def sd_func_school(sX,sY, sids, yearWeighted = True): 
    return np.power(within_school_constantEffect(sX,sY, sids, yearWeighted = yearWeighted), 0.5)


tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_samp100.dta".format(
        spec, droppval)).drop('teachid', axis=1)

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
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())

### Multivariate infeasable regression
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
for row in results.index:
    for col in results.columns:
        results.loc[row, col] = within_school_constantEffect(effects[row], effects[col], sids)
        print('working on {} and {}, result is: {}'.format(row, col, results.loc[row, col]))
XX = results.values.astype('float')
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
for row in results.index:
    for col in results.columns:
        results.loc[row, col] = within_school_constantEffect(effects[row], outcome[col], sids)
xY = results.values.astype('float')
beta = np.linalg.inv(XX).dot(xY)
vary = [within_school_constantEffect(outcome[col], outcome[col], sids) for col in outcome.keys()]
vary_se = [ustat.bsci_varcovar(outcome[col], outcome[col], func = ustat.BB_sd_func, gamma_mat = gamma_draw_mat[:outcome[col].shape[0], :outcome[col].shape[1],:].copy(), ntasks = ncpus) for col in outcome.keys()]
varx = [beta[:,k].T.dot(XX).dot(beta[:,k]) for k in range(beta.shape[1])]


## BB SEs
def BB_within_school_constantEffect(sX,sY, sids, gamma_draw, yearWeighted = True):

    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    sids_ls = []
    for id in tqdm(unique_ids):
        left = sX[(sids == id).max(axis=1)]
        right = sY[(sids == id).max(axis=1)]
        _gamma_draw = gamma_draw[(sids == id).max(axis=1)]

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):

            # Only teachers with some non-missing observations in school "id" for outcomes "X" or "Y"
            if ((~np.isnan(sY[(sids == id)])).sum() > 0) | ((~np.isnan(sY[(sids == id)])).sum() > 0):
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
                sdevs += [ustat.BB_varcovar(left, right, _gamma_draw, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                sids_ls += [id]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within

def BB_multivar(effects, sids, gamma_draw):
    results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
    for row in results.index:
        for col in results.columns:
            results.loc[row, col] = BB_within_school_constantEffect(effects[row], effects[col], sids, gamma_draw[:effects[row].shape[0], :effects[row].shape[1]])
    XX = results.values.astype('float')
    results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
    for row in results.index:
        for col in results.columns:
            results.loc[row, col] = BB_within_school_constantEffect(effects[row], outcome[col], sids, gamma_draw[:effects[row].shape[0], :effects[row].shape[1]])
    xY = results.values.astype('float')
    beta = np.linalg.inv(XX).dot(xY)
    return beta

_gamma = gamma_draw_mat[:effects['Test scores'].shape[0], :effects['Test scores'].shape[1],:].copy()
beta_se = mvar_bsci_varcovar(effects, sids, func = BB_multivar, gamma_mat = _gamma, ntasks = ncpus)



# make table
results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
for r, row in enumerate(results.index):
    for c, col in enumerate(results.columns):
        results.loc[row, col] = "{:4.3f} ({:4.3f})".format(beta[r, c], beta_se[r, c])
        # results.loc[row, col] = "{:4.3f}".format(beta[r, c])
for c, col in enumerate(results.columns):
    results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(vary[c]**0.5, vary_se[c])
    # results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f}".format(vary[c]**0.5)
    results.loc['$R^2$', col] = "{:4.3f}".format(varx[c]/vary[c])


results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlConstant_spec{}_droppval{}_bt{}_longrun_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 

results.loc[:, ['12th grade GPA', 'Graduation', 'College attendance']].to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlConstant_spec{}_droppval{}_bt{}_longrun_academic_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 


results.loc[:, ['Any CJC', 'Criminal arrest', 'Index crime', 'Incarceration']].to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_schlConstant_spec{}_droppval{}_bt{}_longrun_cjc_samp100.tex'.format(spec,droppval,ns), 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
    ) 

















































sys.exit('STOP STOP FINISHED!!!!!!!!! ')

#################################################################################
### 2) Compare within school effects with and without hetero school effects 
#################################################################################

def within_school_constantEffect(sX,sY, sids, yearWeighted = True):

    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    sids_ls = []
    for id in tqdm(unique_ids):
        left = sX[(sids == id).max(axis=1)]
        right = sY[(sids == id).max(axis=1)]

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):

            # Only teachers with some non-missing observations in school "id" for outcomes "X" or "Y"
            if ((~np.isnan(sY[(sids == id)])).sum() > 0) | ((~np.isnan(sY[(sids == id)])).sum() > 0):
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                sids_ls += [id]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within

tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}.dta".format(
        spec, droppval)).drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

sids = tresids.filter(regex='^school_fe', axis=1).values[:,:]

### Covariance with short-run outcomes
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

# First populate SDs
save_sds = {}
for idx, row in enumerate(results.columns):
    print('Working on SD of {}'.format(row), flush=True)
    within_sd = np.power(within_school_constantEffect(effects[row], effects[row], sids), 0.5)
    save_sds[row] = within_sd
    # _bsSE_sd = bsci_school_varcovar(effects[row], effects[row], sids, sd_func_school, nsims=ns, parallel = True, ntasks = cpus)
    # results.loc[row, row] = "{:4.3f} ({:4.3f})".format(within_sd, _bsSE_sd)
    results.loc[row, row] = "{:4.3f} ({:4.3f})".format(within_sd, np.nan)

# Now add correlations
for idx, row in enumerate(results.columns):
    for col in results.iloc[idx+1:].index:
        print('Working on CORR of {} and {}'.format(row, col), flush=True)
        within_cov = within_school_constantEffect(effects[row], effects[col], sids)
        within_corr = within_cov/(save_sds[row] * save_sds[col])
        # _bsSE_corr = bsci_school_varcovar(effects[row], effects[col], sids, correl_func_school, nsims=ns, parallel = True, ntasks = cpus)
        # results.loc[row, col] = "{:4.3f} ({:4.3f})".format(within_corr, _bsSE_corr)
        results.loc[row, col] = "{:4.3f} ({:4.3f})".format(within_corr, np.nan)



#######################################################################
### 1) school FEs as Xs
#######################################################################

df = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_schlFEs_constant_schlFEs_cSet_matlab.dta".format(droppval, spec))
# df = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_schlFEs_constant_schlFEs_cSet.dta".format(spec, droppval))
# df = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}.dta".format(spec, droppval)).drop('teachid', axis=1)
cog = df.filter(regex='^testscores_', axis=1).values[:,:]
crime = df.filter(regex='aoc_crim_r', axis=1).values[:,:]
behave = df.filter(regex='^behavpca_', axis=1).values[:,:]
sids = df.filter(regex='^sid', axis=1).values[:,:]
# sids = df.filter(regex='^school_fe', axis=1).values[:,:]


def within_school_constantEffect(sX,sY, sids, yearWeighted = True):

    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    sids_ls = []
    for id in tqdm(unique_ids):
        left = sX[(sids == id).max(axis=1)]
        right = sY[(sids == id).max(axis=1)]

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):

            # Only teachers with some non-missing observations in school "id" for outcomes "X" or "Y"
            if ((~np.isnan(sY[(sids == id)])).sum() > 0) | ((~np.isnan(sY[(sids == id)])).sum() > 0):
                nteach =  np.sum((sids == id) * (~np.isnan(sX)) * (~np.isnan(sY)))
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                sids_ls += [id]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within


# checking how the sample changes if removing schools with only 1 mover
sids_crime = sids.copy()
sids_crime[np.isnan(behave)] = np.nan
mover = (pd.DataFrame(sids_crime).nunique(axis=1) > 1).values
unique_ids = np.unique(sids_crime[~np.isnan(sids_crime)])

sids_ls = []
num_movers = []
for id in unique_ids:
    num_movers += [np.sum(mover[np.max(sids_crime == id, 1)])]
    sids_ls += [id]





#######################################################################
### 1) Comparison of teacher-year leave-out to teacher-school leave-out
#######################################################################

if False:
    # Load the data
    df_schl = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teachSchl_mean_resids_spec{}_droppval{}.dta".format(
            spec, droppval)).drop('teachid', axis=1)

    df_year = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}.dta".format(
            spec, droppval)).drop('teachid', axis=1)


    def _stds(tresids):
        cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
        math = tresids.filter(regex='^math_', axis=1).values[:,:]
        eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
        study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
        behave = tresids.filter(regex='^behavpca_normalized_r', axis=1).values[:,:]
        crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
        crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
        gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
        college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
        grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

        effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
        results = pd.DataFrame(columns=['SD'], index=effects.keys())

        ## Leave out school rather than year --- var covar
        for idx, row in enumerate(list(effects.keys())):
            print('idx: {}'.format(idx))
            print('row: {}'.format(row))
            _sdev = ustat.sd_func(effects[row], effects[row])
            _lb, _ub, _se = ustat.bsci_varcovar(effects[row], effects[row], ustat.sd_func, nsims=ns)
            results.loc[row, 'SD'] = "{:4.3f} ({:4.3f})".format(_sdev, _se)
        return results

    dp = pd.concat([_stds(df_year), _stds(df_schl)], axis=1)
    dp.columns = ['leave-out teacher-year', 'leave-out teacher-school']

##############################################################################################################################################
### 2) Within-school average variance of teacher effects --- 
# comparison between residuals that assume constant teacher effects to those allowing the teacher effects to vary by school 
##############################################################################################################################################

if False:
    df = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec1_droppval7_wschoolids_hetero.dta")
    cog = df.filter(regex='^testscores_', axis=1).values[:,:]
    crime = df.filter(regex='aoc_crim_r', axis=1).values[:,:]
    behave = df.filter(regex='^behavpca_', axis=1).values[:,:]
    sids = df.filter(regex='^sid', axis=1).values[:,:]


    def within_school(sX,sY, sids, yearWeighted = False):

        sdevs = []
        unique_ids = np.unique(sids[~np.isnan(sids)])
        totler = []
        expecs_left = []
        expecs_right = []
        sids_ls = []
        for id in tqdm(unique_ids):
            left = sX.copy()
            left[sids != id] = np.NaN
            right = sY.copy()
            right[sids != id] = np.NaN

            # ONly use schools with at least two teachers
            if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
                nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                expecs_left += [np.nanmean(left)]
                expecs_right += [np.nanmean(right)]
                sids_ls += [id]

        within = np.nansum(sdevs) / np.nansum(totler)
        total = ustat.varcovar(sX, sY, yearWeighted = yearWeighted)
        between = total - within
        return within, between, total    
        # return within, totler,sdevs, sids_ls


    def _stds(tresids):
        cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
        math = tresids.filter(regex='^math_', axis=1).values[:,:]
        eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
        study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
        behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
        crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
        crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
        gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
        college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
        grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

        sids = tresids.filter(regex='^sid', axis=1).values[:,:]

        effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
        results = pd.DataFrame(columns=['SD'], index=effects.keys())

        ## Leave out school rather than year --- var covar
        for idx, row in enumerate(list(effects.keys())):
            print('idx: {}'.format(idx))
            print('row: {}'.format(row))
            within, between, total = within_school(effects[row], effects[row], sids)
            results.loc[row, 'SD'] = "{:4.5f}".format(within**0.5)
        return results

    df_constant = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant.dta".format(spec, droppval))
    df_hetero = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_hetero.dta".format(spec, droppval))

    dp = pd.concat([_stds(df_constant), _stds(df_hetero)], axis=1)
    dp.columns = ['Constant', 'Hetero']


# experementation... 
# within, totler,sdevs, sids_ls = within_school(crime, crime)
# totler = np.array(totler)
# sdevs = np.array(sdevs)
# sids_ls = np.array(sids_ls)
# for l in [2, 5, 10, 12, 15]:
#     print('\n SD of within-school teacher effects: {}'.format( np.power(np.nansum(sdevs[totler>l]) / np.nansum(totler[totler>l]), 0.5) ))

#     _crime = crime[pd.Series(sids[:,0]).isin(sids_ls[totler > l]).values]
#     print('\n SD of teacher effects across schools for this sub-sample of teachers: {} \n\n'.format(ustat.kssOneWay(_crime, _crime)**0.5 ))



#######################################################################
### 3) Within-school average variance-covariance of teacher effects
#######################################################################

sys.exit('STOP STOP STOP')

def withinOnly_school(sX,sY, sids, yearWeighted = False):
    sdevs = []
    unique_ids = np.unique(sids[~np.isnan(sids)])
    totler = []
    expecs_left = []
    expecs_right = []
    for id in tqdm(unique_ids):
        left = sX.copy()
        left[sids != id] = np.NaN
        right = sY.copy()
        right[sids != id] = np.NaN

        # ONly use schools with at least two teachers
        if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
            nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
            sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
            totler += [nteach]
            expecs_left += [np.nanmean(left)]
            expecs_right += [np.nanmean(right)]

    within = np.nansum(sdevs) / np.nansum(totler)
    return within 

def correl_func_school(sX,sY, sids, yearWeighted = False): 
    return withinOnly_school(sX,sY, sids, yearWeighted = yearWeighted)/np.power(withinOnly_school(sX,sX, sids, yearWeighted = yearWeighted)*withinOnly_school(sY,sY, sids, yearWeighted = yearWeighted), 0.5) 


def sd_func_school(sX,sY, sids, yearWeighted = False): 
    return np.power(withinOnly_school(sX,sY, sids, yearWeighted = yearWeighted), 0.5)


def bsci_school_helper(s,X,Y, sids,func, yearWeighted):
    np.random.seed()
    samp = np.random.choice(X.shape[0],X.shape[0],replace=True)
    sX = X[samp,:]
    sY = Y[samp,:]
    sids = sids[samp,:]
    return func(sX,sY, sids, yearWeighted = yearWeighted)

def bsci_school_varcovar(X,Y, sids, func ,nsims=100,alpha=0.05, yearWeighted = False, parallel = False, ntasks = np.nan):
    '''
    returns alpha/2 and 1-alpha/2 quantiles of SD estimate from nsims bootstrap
    reps sampling teachers with replacement
    '''
    if parallel:
        # check that "ntasks" is defined
        if np.isnan(ntasks):
            sys.exit('Need to specify "ntasks"')

        _foo = partial(bsci_school_helper, X=X, Y=Y, sids = sids, func=func, yearWeighted = yearWeighted)
        if __name__ == '__main__':
            with Pool(ntasks) as p:
                result = list(tqdm(p.imap(_foo, range(nsims)), total=nsims))
    else:
        result = []
        for s in tqdm(range(nsims)):
            samp = np.random.choice(X.shape[0],X.shape[0],replace=True)
            sX = X[samp,:]
            sY = Y[samp,:]
            result += [func(sX,sY, sids, yearWeighted = yearWeighted)]

    # Parse the results 
    rr = np.array(result)
    rr = np.power(np.nanvar(rr), 0.5)
    return rr

if True:
    df_constant = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant.dta".format(spec, droppval))
    df_hetero = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_hetero.dta".format(spec, droppval))

    ns = 100
    cpus = 30


    def _within_varcov(tresids):
        cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
        math = tresids.filter(regex='^math_', axis=1).values[:,:]
        eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
        study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
        behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
        crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
        crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
        gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
        college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
        grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

        sids = tresids.filter(regex='^sid', axis=1).values[:,:]

        ### Covariance with short-run outcomes
        effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
        results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

        # First populate SDs
        save_sds = {}
        for idx, row in enumerate(results.columns):
            print('Working on SD of {}'.format(row), flush=True)
            within_sd = np.power(withinOnly_school(effects[row], effects[row], sids), 0.5)
            save_sds[row] = within_sd
            _bsSE_sd = bsci_school_varcovar(effects[row], effects[row], sids, sd_func_school, nsims=ns, parallel = True, ntasks = cpus)
            results.loc[row, row] = "{:4.3f} ({:4.3f})".format(within_sd, _bsSE_sd)

        # Now add correlations
        for idx, row in enumerate(results.columns):
            for col in results.iloc[idx+1:].index:
                print('Working on CORR of {} and {}'.format(row, col), flush=True)
                within_cov = withinOnly_school(effects[row], effects[col], sids)
                within_corr = within_cov/(save_sds[row] * save_sds[col])
                _bsSE_corr = bsci_school_varcovar(effects[row], effects[col], sids, correl_func_school, nsims=ns, parallel = True, ntasks = cpus)
                results.loc[row, col] = "{:4.3f} ({:4.3f})".format(within_corr, _bsSE_corr)
        return results

    # effects_constant = _within_varcov(df_constant)
    effects_hetero = _within_varcov(df_hetero)
    print('Finished calculations!')

    long_run_vars = ['Any CJC', 'Criminal arrest', '12th grade GPA', 'College attendance', 'Graduation']
    short_run_vars =['Test scores','Behaviors','Study skills']

    ### i) Variance-covariance of long-run effects
    # Constant teacher effects
    # tab = effects_constant.loc[long_run_vars].drop(short_run_vars, axis=1)
    # tab.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/withinSchl_varcovar_spec{}_droppval{}_longrun_constant.tex'.format(spec,droppval), 
    #     na_rep='', escape=False, 
    #     multicolumn_format = 'c',
    #     column_format = 'c' * int(tab.shape[1] + 1)
    #     )  

    # Hetero teacher effects
    tab = effects_hetero.loc[long_run_vars].drop(short_run_vars, axis=1) 
    tab.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/withinSchl_varcovar_spec{}_droppval{}_longrun_hetero.tex'.format(spec,droppval), 
        na_rep='', escape=False, 
        multicolumn_format = 'c',
        column_format = 'c' * int(tab.shape[1] + 1)
        ) 


    # ### ii) Correlation of short- and long-run effects
    # # Constant teacher effects
    # tab = effects_constant.loc[short_run_vars].drop(short_run_vars, axis=1) 
    # tab.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/withinSchl_varcovar_correl_only_spec{}_droppval{}_longrun_constant.tex'.format(spec,droppval), 
    #     na_rep='', escape=False, 
    #     multicolumn_format = 'c',
    #     column_format = 'c' * int(tab.shape[1] + 1)
    #     )  

    # Hetero teacher effects
    tab = effects_hetero.loc[short_run_vars].drop(short_run_vars, axis=1) 
    tab.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/withinSchl_varcovar_correl_only_spec{}_droppval{}_longrun_hetero.tex'.format(spec,droppval), 
        na_rep='', escape=False, 
        multicolumn_format = 'c',
        column_format = 'c' * int(tab.shape[1] + 1)
        ) 

#######################################################################
### 4) Within-school 1SD effects (relative to mean)
#######################################################################

# tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant.dta".format(spec, droppval))
# cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
# study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
# behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
# crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
# crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
# aoc_traff = tresids.filter(regex='aoc_traff_r', axis=1).values[:,:]
# aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
# aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]

# sids = tresids.filter(regex='^sid', axis=1).values[:,:]

# effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Criminal arrest':crime, 'Traffic citation':aoc_traff, 'Index crime':aoc_index, 'Incarceration':aoc_incar}

# results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
# for row in results.index:
#     for idx, col in enumerate(results.columns):
#         if row != col:
#             effect = withinOnly_school(effects[row], effects[col], sids)
#             lb, ub, sd = ustat.bsci_varcovar(effects[row], effects[col], ustat.sd_effect_func, nsims=ns)
#             results.loc[row, col] = (effect, sd)

# means = {'Criminal arrest':0.24, 'Traffic citation':0.35, 'Index crime':0.11, 'Incarceration':0.10}
# for row in labels:
#     for col in means.keys():
#         tmp = results.loc[row, col]
#         tmp = (tmp[0]/means[col]*100,tmp[1]/means[col]*100)
#         results.loc[row, col] = tmp
# fig, axes = plt.subplots(figsize=(8,5))
# x = np.arange(len(labels))
# width = 0.2
# for adj, lab in [(-1.5, 'Criminal arrest'), (-0.5, 'Traffic citation'), (0.5, 'Index crime'), (1.5, 'Incarceration')]:
#     axes.bar(x+adj*width, results.loc[labels, lab].apply(lambda x: x[0]).values,
#             width,
#             yerr=results.loc[labels, lab].apply(lambda x: x[1]).values*1.96,
#             label=lab)
# axes.set_ylabel('% effect of 1 SD increase on long-run outcomes') 
# axes.set_xlabel('Short-run outcome') 
# axes.set_xticks(x)
# axes.set_xticklabels(labels)
# axes.legend()
# fig.tight_layout()
# fig.savefig('/accounts/projects/crwalters/cncrime/teachers_final/figures/sd_effects_spec{}_droppval{}_cjcomponents_normed.pdf'.format(spec,droppval))



#######################################################################
### 5) Within-school infeasable regression
#######################################################################

if True:
    nsims = 100
    ntasks = 32

    def withinOnly_school(sX,sY, sids, yearWeighted = False):
        sdevs = []
        unique_ids = np.unique(sids[~np.isnan(sids)])
        totler = []
        expecs_left = []
        expecs_right = []
        for id in tqdm(unique_ids):
            left = sX.copy()
            left[sids != id] = np.NaN
            right = sY.copy()
            right[sids != id] = np.NaN

            # ONly use schools with at least two teachers
            if (np.sum(np.sum(~np.isnan(left),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(right),1) >= 2) >= 2):
                nteach =  np.sum(np.sum(~np.isnan(left),1) >= 2)
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
                expecs_left += [np.nanmean(left)]
                expecs_right += [np.nanmean(right)]

        within = np.nansum(sdevs) / np.nansum(totler)
        return within 

    if True:
        ### Multivariate regressions --- decmposing teacher effects on crime to explained (and unexplained) parts by short-run teacher measured effects
        # tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant.dta".format(spec, droppval))
        tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_hetero.dta".format(spec, droppval))

        cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
        math = tresids.filter(regex='^math_', axis=1).values[:,:]
        eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
        study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
        behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
        crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
        crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
        gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
        college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
        grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

        sids = tresids.filter(regex='^sid', axis=1).values[:,:]

        # Regression decomposition
        effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study}
        outcome = {'Any CJC':crimeany,'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college}

        results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
        for row in results.index:
            for col in results.columns:
                results.loc[row, col] = withinOnly_school(effects[row], effects[col], sids)
        XX = results.values.astype('float')
        results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
        for row in results.index:
            for col in results.columns:
                results.loc[row, col] = withinOnly_school(effects[row], outcome[col], sids)
        xY = results.values.astype('float')
        beta = np.linalg.inv(XX).dot(xY)
        vary = [withinOnly_school(outcome[col], outcome[col], sids) for col in outcome.keys()]
        varx = [beta[:,k].T.dot(XX).dot(beta[:,k]) for k in range(beta.shape[1])]

        # boot strap SE
        def _bs_reg(s):
            samp = np.random.choice(cog.shape[0], cog.shape[0], replace=True)
            results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
            for row in results.index:
                for col in results.columns:
                    results.loc[row, col] = withinOnly_school(effects[row][samp,:], effects[col][samp,:], sids[samp,:])
            XX = results.values.astype('float')
            results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
            for row in results.index:
                for col in results.columns:
                    results.loc[row, col] = withinOnly_school(effects[row][samp,:], outcome[col][samp,:], sids[samp,:])
            xY = results.values.astype('float')
            return np.linalg.inv(XX).dot(xY)

        if __name__ == '__main__':
            with Pool(ntasks) as p:
                se_result = list(tqdm(p.imap(_bs_reg, range(nsims)), total=nsims))

            
        # se_result = []
        # for s in tqdm(range(ns)):
        #     samp = np.random.choice(cog.shape[0], cog.shape[0], replace=True)
        #     results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
        #     for row in results.index:
        #         for col in results.columns:
        #             results.loc[row, col] = withinOnly_school(effects[row][samp,:], effects[col][samp,:], sids[samp,:])
        #     XX = results.values.astype('float')
        #     results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
        #     for row in results.index:
        #         for col in results.columns:
        #             results.loc[row, col] = withinOnly_school(effects[row][samp,:], outcome[col][samp,:], sids[samp,:])
        #     xY = results.values.astype('float')
        #     se_result += [np.linalg.inv(XX).dot(xY)]

        # get SEs of SD of outcomes
        se_sdy = []
        for col in list(outcome.keys()):
            _sesd = bsci_school_varcovar(outcome[col], outcome[col], sids, sd_func_school, nsims=ns, parallel = True, ntasks = ntasks)
            se_sdy += [_sesd]

        # make table
        results = pd.DataFrame(columns=outcome.keys(), index=effects.keys())
        for r, row in enumerate(results.index):
            for c, col in enumerate(results.columns):
                results.loc[row, col] = "{:4.3f} ({:4.3f})".format(beta[r, c], 
                        np.var([k[r,c] for k in se_result])**0.5)
        for c, col in enumerate(results.columns):
            results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(vary[c]**0.5, se_sdy[c])
            results.loc['$R^2$', col] = "{:4.3}".format(varx[c]/vary[c])

        results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_spec{}_droppval{}_longrun_wschool_hetero.tex'.format(spec,droppval),  
            na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
            ) 

        # results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/shortrun_mvreg_spec{}_droppval{}_longrun_wschool_constant.tex'.format(spec,droppval),  
        #     na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(results.shape[1] + 1)
        #     ) 

#######################################################################
### 6) Within-school using Matlab connected set
#######################################################################


# # matlab_data = pd.read_csv('/accounts/projects/crwalters/cncrime/teachers_final/code/KSS/LeaveOutTwoWay-evan/leave_out_estimates.csv', sep="\t") # This has match-level connected set, gives weired results for U-stats
# matlab_data = pd.read_csv('/accounts/projects/crwalters/cncrime/teachers_final/code/KSS/LeaveOutTwoWay-evan/leave_out_estimates.csv', sep="\t")
# matlab_data.columns = ['testscores_r','teachid','school_fe','leverage_matlab']
# matlab_data['teach_school'] =matlab_data.groupby(['teachid','school_fe']).grouper.group_info[0]
# matlab_data['_tmp']=1
# matlab_data['obs'] =matlab_data.groupby(['teach_school'])['_tmp'].cumsum() - 1
# matlab_cog = np.array(sparse.csr_matrix((matlab_data.testscores_r, (matlab_data.teach_school, matlab_data.obs))).todense())
# matlab_sids = np.array(sparse.csr_matrix((matlab_data.school_fe, (matlab_data.teach_school, matlab_data.obs))).todense())

# sd_within = withinOnly_school(matlab_cog, matlab_cog, matlab_sids)**0.5


# matlab_data['teachid'] =matlab_data.groupby(['teachid']).grouper.group_info[0]
# matlab_data['_tmp']=1
# matlab_data['obs'] =matlab_data.groupby(['teachid'])['_tmp'].cumsum() - 1
# matlab_cog = np.array(sparse.csr_matrix((matlab_data.testscores_r, (matlab_data.teachid, matlab_data.obs))).todense())

# sd_all = ustat.varcovar(matlab_cog, matlab_cog)**0.5


###############################################################################################
### 7) Within-school using \Gamma estimated using full data vs. connected set
###############################################################################################

# spec = 7
# droppval = 1
# ns = 100

# # Load the data
# GammaFull = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant.dta".format(spec, droppval))
# GammaCS = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_wschoolids_constant_connectedSet.dta".format(spec, droppval))

# def fun_sds(tresids, label):
#     cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
#     math = tresids.filter(regex='^math_', axis=1).values[:,:]
#     eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
#     study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
#     behave = tresids.filter(regex='^behavpca_r', axis=1).values[:,:]
#     crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
#     crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
#     incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]
#     gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
#     college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
#     grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

#     sids = tresids.filter(regex='^sid', axis=1).values[:,:]

#     ### Covariance with short-run outcomes
#     effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
#     results = pd.DataFrame(columns=[label], index=effects.keys())

#     for idx, row in enumerate(list(effects.keys())):
#         print('idx: {}'.format(idx))
#         print('row: {}'.format(row))
#         results.loc[row, label] = sd_func_school(effects[row], effects[row], sids = sids)

#     return results

# _full = fun_sds(GammaFull, label = 'Full sample')
# _cs = fun_sds(GammaCS, label = 'Connected set')
# tab = pd.concat([_full, _cs], axis=1)

# tab.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/stds_wschool_csComparison_spec{}_droppval{}.tex'.format(spec,droppval), 
#     na_rep='', escape=False, 
#     multicolumn_format = 'c',
#     column_format = 'c' * int(tab.shape[1] + 1)
# )  




sys.exit('STOP STOP STOP')





## 
tresids = pd.read_stata("/accounts/projects/crwalters/cncrime/teachers_final/dump/teach_mean_resids_spec{}_droppval{}_schlFEs_constant.dta".format(spec, droppval))
X = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
sids = tresids.filter(regex='^sid', axis=1).values[:,:]
ustat.sd_func(X, X)
sd_func_school(X, X, sids)







### Simulation estimator (VARIANCE)
outcome = 'testscores'
df = tresids.filter(regex='(^testscores_r|^aoc_crim_r|^sid)', axis=1)
df['teachid'] = df.index
df = pd.wide_to_long(df, stubnames = ['testscores_r','aoc_crim_r','sid'], i = 'teachid', j = 't').reset_index()
df.sort_values(['teachid','sid','t'], inplace=True)



df = df.loc[df.aoc_crim_r.notnull() & df.testscores_r.notnull()]

itr = 1
cont = True
while cont:
    print('cont = {}, Iteration {}'.format(cont, itr))
    _tmp = df.shape[0]
    df = df.loc[df.groupby('teachid')['t'].transform('nunique') > 1]    
    df = df.loc[df.groupby('sid')['teachid'].transform('nunique') > 1]
    cont = (df.shape[0] != _tmp)
    itr += 1


def randomMatching_varcovar(Nsims = 50):
    np.random.seed()
    rr = np.zeros((Nsims,1))
    for cc in range(Nsims):
        df['_sim'] = np.random.normal(size=df.shape[0])
        df.sort_values(['teachid','_sim'], inplace=True)
        df['_n'] = 1
        df['_n'] = df.groupby(['teachid'])['_n'].transform('cumsum') 
        dp = df.loc[df['_n'] <= 2].copy()
        dp = dp.pivot_table(index = ['teachid','sid'], values = ['X','Y'], columns = ['_n']).reset_index()
        dp.columns = ['teachid','sid','X1','X2','Y1','Y2']

        dp.groupby('sid')[['X1','Y2']].apply(lambda z: np.cov(z['X1'],z['Y2'])[0,1])




        rr[cc,0] = np.cov(df.loc[df['_n']==1, outcome+'_r'], df.loc[df['_n']==2, outcome+'_r'])[0,1]
    return np.mean(rr)













withinOnly_school(effects[row], effects[col], sids)/np.power(withinOnly_school(effects[row], effects[row], sids) * withinOnly_school(effects[col], effects[col], sids), 0.5)
















sys.exit('STOP STOP STOP')




# ### Covariance with short-run outcomes
# effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
# results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

# # First populate SDs
# for idx, row in enumerate(results.columns):
#     sdev = ustat.sd_func(effects[row], effects[row])
#     # lb, ub, se = ustat.bsci_varcovar(effects[row], effects[row], ustat.sd_func, nsims=ns)
#     # results.loc[row, row] = "{:4.3f} ({:4.3f})".format(sdev, se)
#     results.loc[row, row] = "{:4.3f}".format(sdev)

# # Now add correlations
# for idx, row in enumerate(results.columns):
#     for col in results.iloc[idx+1:].index:
#         sdev = ustat.correl_func(effects[row], effects[col])
#         # lb, ub, se = ustat.bsci_varcovar(effects[row], effects[col], ustat.correl_func, nsims=ns)
#         results.loc[row, col] = "{:4.3f}".format(sdev)

# # Save to latex
# results.to_latex('/accounts/projects/crwalters/cncrime/teachers_final/tables/varcovar_spec{}_droppval{}_longrun.tex'.format(spec,droppval), 
#     na_rep='', escape=False, 
#     multicolumn_format = 'c',
#     column_format = 'c' * int(results.shape[1] + 1)
#     )  


# results.loc[['Test scores','Behaviors','Study skills']].drop(['Test scores','Behaviors','Study skills'],axis=1)








#######################################################################





# ustat.sd_func(behave, behave)
# ustat.bsci_varcovar(behave, behave, ustat.sd_func, nsims=ns)

# behave = df_year.filter(regex='^behavpca_r', axis=1).values[:,:]
# crime = df_year.filter(regex='^aoc_crim_r', axis=1).values[:,:]
# # ustat.sd_func(behave, behave)
# # check(crime, crime, ustat.sd_func)

# # ustat.bsci_varcovar(crime,crime,func = ustat.sd_func, nsims=10)

# ustat.bsci_varcovar(crime,crime,func = ustat.sd_func, nsims=10, parallel = True, ntasks = 8)





# def bsci_varcovar(X,Y,func,nsims=10,alpha=0.05, yearWeighted = False, parallel = False, ntasks = np.nan):
#     '''
#     returns alpha/2 and 1-alpha/2 quantiles of SD estimate from nsims bootstrap
#     reps sampling teachers with replacement
#     '''
#     if parallel:
#         # check that "ntasks" is defined
#         if np.isnan(ntasks):
#             sys.exit('Need to specify "ntasks"')

#         _foo = partial(ustat.bsci_helper, X=X, Y=Y, func=func, yearWeighted = yearWeighted)
#         with Pool(ntasks) as p:
#             result = list(tqdm(p.imap(_foo, range(nsims)), total=nsims))
#     else:
#         result = []
#         for s in tqdm(range(nsims)):
#             samp = np.random.choice(X.shape[0],X.shape[0],replace=True)
#             sX = X[samp,:]
#             sY = Y[samp,:]
#             result += [func(sX,sY, yearWeighted = yearWeighted)]

#     # Parse the results (allowing the output of "func" to be a tuple of more than one element)
#     if len(result[0]) > 1:
#         result = np.array(result)
#         rr = {}
#         for l in range(result.shape[1]):
#             rr{'lb{}'.format(l+1)} = np.quantile(result[:,l],alpha/2)
#             rr{'ub{}'.format(l+1)} = np.quantile(result[:,l],1-alpha/2)
#             rr{'sd{}'.format(l+1)} = np.nanvar(result[:,l])**0.5
#         return rr
#     else:
#         lb = np.quantile(result,alpha/2)
#         ub = np.quantile(result,1-alpha/2)
#         return lb, ub, np.nanvar(result)**0.5
