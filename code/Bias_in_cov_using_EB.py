"""
Bias_in_cov_using_EB.py — Figure B.1 and Table B.1 (Appendix B)

Appendix B asks whether covariances of Empirical Bayes (EB) posterior means
across outcomes recover covariances of latent teacher effects. It shows that,
because of shrinkage and correlated classroom-level sampling error, they do not.

This script:
  1. Estimates the variance-covariance of latent teacher effects (U-statistics)
     and of classroom-level sampling error for test scores, behaviors, and
     criminal arrest, and writes Table B.1 (SDs on the diagonal, correlations
     off-diagonal, with sampling-error analogues in brackets).
  2. Uses those estimated moments to compute, in closed form, the ratio between
     the covariance (or correlation) of multivariate EB posteriors and the
     covariance (or correlation) of latent effects, for a grid of class sizes n,
     producing the four panels of Figure B.1.

The EB computation is deterministic (closed-form linear algebra over the
estimated moments); the seed below is kept only for consistency with the other
scripts in this archive.

Reads:  temp/teach_mean_resids.dta   (wide teacher-year residuals from estimate_variance.do)
Writes: tables/tableB1.tex
        figures/figureB1a.pdf   Panel A — covariances, zero correlated sampling error
        figures/figureB1b.pdf   Panel B — covariances, no restrictions
        figures/figureB1c.pdf   Panel C — correlations, zero correlated sampling error
        figures/figureB1d.pdf   Panel D — correlations, no restrictions

NOTE: As with every output in this archive, this runs on the SIMULATED data and
will NOT reproduce the estimates in the paper. In particular, the simulated data
place each teacher in only two class-years, so the classroom-level sampling-error
moments and the average class size (avg_n) are estimated off very little
within-teacher variation and the resulting figure/table can look degenerate.

For reference, the published Table B.1 (real data) reports:
                    Test scores      Behaviors        Criminal arrest
    Test scores     0.121 [0.1391]   0.056 [0.0614]   -0.008 [-0.0404]
    Behaviors                        0.125 [0.2144]   -0.202 [-0.1048]
    Criminal arrest                                    0.027 [0.0697]

This file was originally run on the EML server; it has been adapted to run
against the simulated data shipped with this archive (server paths removed,
inputs read from temp/, outputs written to figures/ and tables/).
"""

import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys
from multiprocessing import Pool

# import ustat functions (code/ is on sys.path when run as `python code/<file>.py`)
import funcs_vcov_ustats as ustat

# Graphing
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import norm


#####################################
### 0) Options/globals
#####################################
np.random.seed(93293483)


#####################################
### 1) Load data
#####################################
# Wide teacher-year residuals produced by estimate_variance.do. Each outcome has
# one column per within-teacher observation index (e.g. testscores_r1, ...).
tresids = pd.read_stata("temp/teach_mean_resids.dta").drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:, :]
math = tresids.filter(regex='^math_', axis=1).values[:, :]
eng = tresids.filter(regex='^eng_', axis=1).values[:, :]
study = tresids.filter(regex='^studypca_', axis=1).values[:, :]
behave = tresids.filter(regex='^behave_r', axis=1).values[:, :]

gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:, :]
college = tresids.filter(regex='college_bound_', axis=1).values[:, :]
grad = tresids.filter(regex='^grad_', axis=1).values[:, :]

crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:, :]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:, :]
aoc_traff = tresids.filter(regex='aoc_traff_r', axis=1).values[:, :]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:, :]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:, :]

obs = tresids.filter(regex='^obs_cog', axis=1).values[:, :]
avg_n = np.nanmean(obs)

#####################################
### 2) Estimation
#####################################


def estEB_fun(nlist, origX, origY, no_corr_errors=False):

	### 1) Variance-covariance of teacher effects
	varx = ustat.varcovar(origX, origX)
	vary = ustat.varcovar(origY, origY)
	covar = ustat.varcovar(origY, origX)
	corrxy = covar/np.power(varx*vary, 0.5)


	### 2) Variance-covariance of sampling/measurment error
	# Within-teacher residuals
	# Means (over teachers)
	mux = np.nanmean(np.nanmean(origX, axis=1))
	muy = np.nanmean(np.nanmean(origY, axis=1))

	origX_within = origX - np.nanmean(origX,1)[:,None]
	origY_within = origY - np.nanmean(origY,1)[:,None]

	origX_within_var = pd.DataFrame({'a':origX_within.ravel(), 'b':origX_within.ravel()}).cov().values[0,1]
	origY_within_var = pd.DataFrame({'a':origY_within.ravel(), 'b':origY_within.ravel()}).cov().values[0,1]
	if no_corr_errors :
		origXY_within_covar = 0
	else:
		origXY_within_covar = pd.DataFrame({'a':origX_within.ravel(), 'b':origY_within.ravel()}).cov().values[0,1]

	### 2b) Multiply within variance/covariance by the average number of students per teacher
	origXY_within_covar = origXY_within_covar*avg_n
	origX_within_var = origX_within_var*avg_n
	origY_within_var = origY_within_var*avg_n

	### 3) EB shrinkage
	CovEB = []
	CorrEB = []
	for n in nlist:
		print("Working on n = {} \n".format(n))
		# a) Matrix of variance-covariance of the vector (\bar{Y}^X, \bar{Y}^Y) with number of obs in each outcome of "n"
		Sigma_vars = np.array([
			[ varx +  origX_within_var/n , covar + origXY_within_covar/n],
			[covar + origXY_within_covar/n, vary + origY_within_var/n]
			])

		# b) Vector of covariance of teacher true effects on X (or Y) with (\bar{Y}^X, \bar{Y}^Y)
		Sigma_covs_X = np.array([varx, covar ]).reshape(1,-1)
		Sigma_covs_Y = np.array([covar, vary ]).reshape(1,-1)

		# Shrinkage coefs:
		lambdaX = Sigma_covs_X.dot(np.linalg.inv(Sigma_vars))
		lambdaY = Sigma_covs_Y.dot(np.linalg.inv(Sigma_vars))

		lambdaX_var = np.power(lambdaX[0,0],2)*Sigma_vars[0,0] + np.power(lambdaX[0,1],2)*Sigma_vars[1,1] + 2*lambdaX[0,0]*lambdaX[0,1]*Sigma_vars[0,1]
		lambdaY_var = np.power(lambdaY[0,0],2)*Sigma_vars[0,0] + np.power(lambdaY[0,1],2)*Sigma_vars[1,1] + 2*lambdaY[0,0]*lambdaY[0,1]*Sigma_vars[0,1]

		# c) Covariance of EB estimates of \alpha^X_EB and \alpha^Y_EB (\alpha^X_EB = E[\alpha^X |\bar{Y}^X, \bar{Y}^Y ] = \lambdaX[0] \bar{Y}^X + \lambdaX[1] \bar{Y}^Y )
		# Cov(EB estimates) = \lambdaX[0]*\lambdaY[0] Var(\bar{Y}^X,)
		#					+ \lambdaX[0]*\lambdaY[1] Cov(\bar{Y}^X, \bar{Y}^Y)
		#					+ \lambdaX[1]*\lambdaY[0] Cov(\bar{Y}^X, \bar{Y}^Y)
		#					+ \lambdaX[1]*\lambdaY[1] Var(\bar{Y}^Y)

		_CovEB = lambdaX[0,0]*lambdaY[0,0]*Sigma_vars[0,0] + lambdaX[0,0]*lambdaY[0,1]*Sigma_vars[0,1] + lambdaX[0,1]*lambdaY[0,0]*Sigma_vars[0,1] + lambdaX[0,1]*lambdaY[0,1]*Sigma_vars[1,1]

		CovEB += [_CovEB, ]
		CorrEB += [_CovEB/np.power(lambdaX_var*lambdaY_var, 0.5),]

	# Prepare results for export
	rr = pd.DataFrame({'CorrEB':CorrEB, 'CovEB':CovEB,  'n':nlist})
	rr['true_corrxy'] = corrxy
	rr['true_covarxy'] = covar
	rr['dev_true_corr'] = rr['CorrEB']/rr['true_corrxy']
	rr['dev_true_cov'] = rr['CovEB']/rr['true_covarxy']

	return rr



n_vec = np.arange(10, 1000, 5).tolist()

# EB posterior bias, using the estimated (possibly correlated) sampling error
rr_behave_crime = estEB_fun(n_vec, behave, crime)
rr_cog_crime = estEB_fun(n_vec, cog, crime)
rr_cog_behave = estEB_fun(n_vec, cog, behave)


# EB posterior bias, imposing no correlated measurement/sampling errors
rr_behave_crime0 = estEB_fun(n_vec, behave, crime, True)
rr_cog_crime0 = estEB_fun(n_vec, cog, crime, True)
rr_cog_behave0 = estEB_fun(n_vec, cog, behave, True)


#####################################
# Make figures (Figure B.1, Panels A-D)
#####################################
plt.rc('text', usetex=False)


####### Panel C — Correlations, imposing zero correlated sampling error #######
fig, axes = plt.subplots()
axes.plot(rr_cog_crime0['n'] ,rr_cog_crime0['dev_true_corr'], 'b' + 'D' +'-', alpha=0.7, label='Corr(Crime effects, Test scores effects)')
axes.plot(rr_cog_behave0['n'] ,rr_cog_behave0['dev_true_corr'], 'g' + 's' +'-', alpha=0.7, label='Corr(Test scores effects, Behavior effects)')
axes.plot(rr_behave_crime0['n'] ,rr_behave_crime0['dev_true_corr'], 'r' + 'o' +':', fillstyle='none', alpha=0.7, label='Corr(Crime effects, Behavior effects)')
axes.hlines(y = 1, xmin = -100, xmax=10000, color = 'black', label='Estimated Corr. equals true one')
axes.set_xlim([float(rr_cog_crime0['n'].min()-20), float(rr_cog_crime0['n'].max()+10)])
axes.set_xlabel('Number of students in classroom', fontsize=12)
axes.set_ylabel('estimate/true Corr. \n in latent teacher effects', fontsize=12)
axes.legend(fontsize='medium', loc='upper right')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
axes.set_facecolor('white')
plt.grid(axis = 'x')
fig.tight_layout()
fig.savefig('figures/figureB1c.pdf')


####### Panel D — Correlations, no restrictions #######
fig, axes = plt.subplots()
axes.plot(rr_cog_crime['n'] ,rr_cog_crime['dev_true_corr'], 'b' + 'D' +'-', alpha=0.7, label='Corr(Crime effects, Test scores effects)')
axes.plot(rr_cog_behave['n'] ,rr_cog_behave['dev_true_corr'], 'g' + 's' +'-', alpha=0.7, label='Corr(Test scores effects, Behavior effects)')
axes.plot(rr_behave_crime['n'] ,rr_behave_crime['dev_true_corr'], 'r' + 'o' +':', fillstyle='none', alpha=0.7, label='Corr(Crime effects, Behavior effects)')
axes.hlines(y = 1, xmin = -100, xmax=10000, color = 'black', label='Estimated Corr. equals true one')
axes.set_xlim([float(rr_cog_crime['n'].min()-20), float(rr_cog_crime['n'].max()+10)])
axes.set_xlabel('Number of students in classroom', fontsize=12)
axes.set_ylabel('estimate/true Corr. \n in latent teacher effects', fontsize=12)
axes.legend(fontsize='medium', loc='lower right')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
axes.set_facecolor('white')
plt.grid(axis = 'x')
fig.tight_layout()
fig.savefig('figures/figureB1d.pdf')


####### Panel A — Covariances, imposing zero correlated sampling error #######
fig, axes = plt.subplots()
axes.plot(rr_cog_crime0['n'] ,rr_cog_crime0['dev_true_cov'], 'b' + 'D' +'-', alpha=0.7, label='Cov(Crime effects, Test scores effects)')
axes.plot(rr_cog_behave0['n'] ,rr_cog_behave0['dev_true_cov'], 'g' + 's' +'-', alpha=0.7, label='Cov(Test scores effects, Behavior effects)')
axes.plot(rr_behave_crime0['n'] ,rr_behave_crime0['dev_true_cov'], 'r' + 'o' +':', fillstyle='none', alpha=0.7, label='Cov(Crime effects, Behavior effects)')
axes.hlines(y = 1, xmin = -100, xmax=10000, color = 'black', label='Estimated Cov. equals true one')
axes.set_xlim([float(rr_cog_crime0['n'].min()-20), float(rr_cog_crime0['n'].max()+10)])
axes.set_xlabel('Number of students in classroom', fontsize=12)
axes.set_ylabel('estimate/true Cov. \n in latent teacher effects', fontsize=12)
axes.legend(fontsize='medium', loc='lower right')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
axes.set_facecolor('white')
plt.grid(axis = 'x')
fig.tight_layout()
fig.savefig('figures/figureB1a.pdf')


####### Panel B — Covariances, no restrictions #######
fig, axes = plt.subplots()
axes.plot(rr_cog_crime['n'] ,rr_cog_crime['dev_true_cov'], 'b' + 'D' +'-', alpha=0.7, label='Cov(Crime effects, Test scores effects)')
axes.plot(rr_cog_behave['n'] ,rr_cog_behave['dev_true_cov'], 'g' + 's' +'-', alpha=0.7, label='Cov(Test scores effects, Behavior effects)')
axes.plot(rr_behave_crime['n'] ,rr_behave_crime['dev_true_cov'], 'r' + 'o' +':', fillstyle='none', alpha=0.7, label='Cov(Crime effects, Behavior effects)')
axes.hlines(y = 1, xmin = -100, xmax=10000, color = 'black', label='Estimated Cov. equals true one')
axes.set_xlim([float(rr_cog_crime['n'].min()-20), float(rr_cog_crime['n'].max()+10)])
axes.set_xlabel('Number of students in classroom', fontsize=12)
axes.set_ylabel('estimate/true Cov. \n in latent teacher effects', fontsize=12)
axes.legend(fontsize='medium', loc='lower right')
plt.tick_params(axis='x', labelsize=12)
plt.tick_params(axis='y', labelsize=12)
axes.set_facecolor('white')
plt.grid(axis = 'x')
fig.tight_layout()
fig.savefig('figures/figureB1b.pdf')


##########################################################################################
# Table B.1: var-cov of true effects and measurement errors across outcomes
##########################################################################################


def sd_func_errors(origX):

	# Within-teacher residuals
	# Means (over teachers)
	mux = np.nanmean(np.nanmean(origX, axis=1))

	origX_within = origX - np.nanmean(origX,1)[:,None]

	origX_within_var = pd.DataFrame({'a':origX_within.ravel(), 'b':origX_within.ravel()}).cov().values[0,1]

	return np.power(origX_within_var, 0.5)

def correl_func_errors(origX,origY):

	# Within-teacher residuals
	# Means (over teachers)
	mux = np.nanmean(np.nanmean(origX, axis=1))
	muy = np.nanmean(np.nanmean(origY, axis=1))

	origX_within = origX - np.nanmean(origX,1)[:,None]
	origY_within = origY - np.nanmean(origY,1)[:,None]

	origX_within_var = pd.DataFrame({'a':origX_within.ravel(), 'b':origX_within.ravel()}).cov().values[0,1]
	origY_within_var = pd.DataFrame({'a':origY_within.ravel(), 'b':origY_within.ravel()}).cov().values[0,1]
	origXY_within_covar = pd.DataFrame({'a':origX_within.ravel(), 'b':origY_within.ravel()}).cov().values[0,1]

	return origXY_within_covar/np.power(origX_within_var*origY_within_var, 0.5)


effects = {'Test scores':cog, 'Behaviors':behave, 'Criminal arrest':crime}
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

## Main var covar
# First populate SDs
for idx, row in enumerate(results.columns):

    # True effects
    sdev = ustat.sd_func(effects[row])

    # Measurment error
    sdev_errors = sd_func_errors(effects[row])

    results.loc[row, row] = "{:4.3f} [{:5.4f}]".format(sdev, sdev_errors)
    print("{:4.3f} [{:5.4f}]".format(sdev, sdev_errors))

# Now add correlations
for idx, row in enumerate(results.columns):
    for col in results.iloc[idx+1:].index:
    	# True effects
    	sdev = ustat.correl_func(effects[row], effects[col])

    	# Measurment error
    	sdev_errors = correl_func_errors(effects[row], effects[col])

    	results.loc[row, col] = "{:4.3f} [{:5.4f}]".format(sdev, sdev_errors)
    	print("{:4.3f} [{:5.4f}]".format(sdev, sdev_errors))


results.to_latex('tables/tableB1.tex',
    na_rep='', escape=False,
    multicolumn_format = 'c',
    column_format = 'c' * int(results.shape[1] + 1)
    )
