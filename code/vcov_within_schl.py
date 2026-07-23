"""
vcov_within_schl.py — Table A9

Within-school variance decomposition. Computes the implied multivariate
regression of long-run teacher effects on short-run effects, restricting
to variation *within* schools. This tests whether the cross-outcome
relationship is driven by between-school sorting.

withinOnly_school(X, Y, sids): for each school, computes the U-statistic
variance/covariance of teacher effects using only teachers within that
school, then aggregates across schools weighted by number of teachers.

tabfunc(): same structure as vcov_implied_reg.py but calls withinOnly_school
instead of the standard varcovar(). SEs via parametric bootstrap.

Reads: temp/teach_mean_resids.dta (includes school_fe columns)
Writes: tables/tableA9.tex
"""

import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys

# For parallel processing
from multiprocessing import Pool
from functools import partial
import funcs_vcov_ustats as ustat

# Seed the parametric bootstrap so the reported SEs are reproducible
np.random.seed(93293483)

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
                # Weight by teachers usable for THIS moment: at least two
                # observed years on each of the two outcomes (for a variance,
                # left == right, so this reduces to the original count)
                nteach =  np.sum((np.sum(~np.isnan(left),1) >= 2) & (np.sum(~np.isnan(right),1) >= 2))
            try:
                sdevs_ses += [ustat.vcv_samp_covar(left, right)*nteach**2]
                sdevs += [ustat.varcovar(left, right, yearWeighted = yearWeighted)*nteach]
                totler += [nteach]
            except:     # If not enough obs to compute SE, skip it
                pass

    within = np.nansum(sdevs) / np.nansum(totler)
    within_se = np.nansum(sdevs_ses) / np.nansum(totler)**2
    return within, within_se

def withinOnly_school_crosscov(sA, sB, sC, sD, sids):
    """
    Sampling covariance between the within-school estimates of Cov(A,B)
    and Cov(C,D). Schools are treated as independent, so
        Cov = sum_s w_s^AB * w_s^CD * Cov(U_s^AB, U_s^CD) / (W^AB * W^CD),
    with the same school eligibility rule and teacher weights w_s as
    withinOnly_school, and the per-school moment sampling covariance from
    ustat.ustat_samp_covar on the school-masked arrays.
    """
    unique_ids = np.unique(sids[~np.isnan(sids)])
    num = 0.0
    W_ab = 0.0
    W_cd = 0.0
    for id in unique_ids:
        mats = []
        for s in (sA, sB, sC, sD):
            m = s.copy()
            m[sids != id] = np.nan
            mats += [m]
        A, B, C, D = mats

        elig_ab = (np.sum(np.sum(~np.isnan(A),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(B),1) >= 2) >= 2)
        elig_cd = (np.sum(np.sum(~np.isnan(C),1) >= 2) >= 2) & (np.sum(np.sum(~np.isnan(D),1) >= 2) >= 2)
        w_ab = np.sum((np.sum(~np.isnan(A),1) >= 2) & (np.sum(~np.isnan(B),1) >= 2)) if elig_ab else 0
        w_cd = np.sum((np.sum(~np.isnan(C),1) >= 2) & (np.sum(~np.isnan(D),1) >= 2)) if elig_cd else 0
        W_ab += w_ab
        W_cd += w_cd

        if elig_ab and elig_cd:
            # Restrict to this school's teachers (rows with any data) --
            # numerically identical, since all-NaN teachers get zero
            # weight in the U-statistic machinery, but much faster.
            keep = ~(np.isnan(A).all(1) & np.isnan(B).all(1) & np.isnan(C).all(1) & np.isnan(D).all(1))
            try:
                num += w_ab * w_cd * ustat.ustat_samp_covar(A[keep], B[keep], C[keep], D[keep])
            except:
                pass
    if W_ab == 0 or W_cd == 0:
        return np.nan
    return num / (W_ab * W_cd)

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
    # Draw the moment vector from a normal with the FULL estimated sampling
    # variance-covariance structure (see the inference appendix: "with
    # sampling variance-covariance structure given by estimated sampling
    # variance-covariances"), not a diagonal approximation.
    mu = results.values.astype(float).ravel()
    combined = {**effects,**outcome}
    row_keys = list(effects.keys())
    col_keys = list(effects.keys()) + list(outcome.keys())
    moms = [(effects[r], combined[c]) for r in row_keys for c in col_keys]

    nm = len(moms)
    var = np.zeros((nm, nm))
    np.fill_diagonal(var, results_vcv.values.astype(float).ravel())
    for i in range(nm):
        for j in range(i + 1, nm):
            cc = withinOnly_school_crosscov(moms[i][0], moms[i][1], moms[j][0], moms[j][1], sids)
            if np.isfinite(cc):
                var[i, j] = cc
                var[j, i] = cc

    # Draw from the eigendecomposition directly: newres = mu + E sqrt(D) z
    # with z ~ N(0, I). This is the same multivariate normal but avoids
    # np.random.multivariate_normal's internal SVD, which can fail to
    # converge on this matrix -- it is EXACTLY singular by construction
    # (the symmetric XX moments appear twice and are perfectly correlated)
    # and its entries span many orders of magnitude. Negative eigenvalues
    # from plug-in estimation noise are clipped to zero.
    var = (var + var.T) / 2
    eigval, eigvec = np.linalg.eigh(var)
    scale = eigvec * np.sqrt(np.clip(eigval, 0, None))

    ns = 500
    bs_ests = np.zeros(shape=(beta.shape[0],beta.shape[1],ns))
    for n in range(ns):
        newres = (mu + scale.dot(np.random.normal(size=len(mu)))).reshape(results.shape)
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
        # Delta method for the SD: Var(sqrt(V)) = Var(V) / (4V)
        sd_se = (se / (4 * within)) ** 0.5 if within > 0 else np.nan
        results.loc['\hline $sd(\mu_j^y)$', col] = "{:4.3f} ({:4.3f})".format(
                within**0.5, sd_se)
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


