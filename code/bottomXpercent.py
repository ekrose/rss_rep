#teacher ustat
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys
from multiprocessing import Pool
import funcs_vcov_ustats as ustat

# Graphing
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'figure.autolayout': True})
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="white", color_codes=True)
plt.style.use('ggplot')

# from scipy.stats import multivariate_normal
from scipy.stats import norm


#####################################
### 0) Options/globals
##################################### 
# Options
ns = 300
ncpus = 21

np.random.seed(93293483)
gamma_draw_mat = np.random.exponential(scale=1, size = (40000,17,ns))


# Color options
col_oracal = 'darkred'
col_SRoracal = 'red'
col_ebSRoracal = 'orange'

# Line style options
lstl_oracal = 'dashed'
lstl_SRoracal = 'dotted'
lstl_ebSRoracal = 'solid'

# Point options
cog_marker = '^'
behave_marker = 's'
study_marker = 'o'

cog_alpha = 0.5
behave_alpha = 0.5
study_alpha = 0.5

cog_col = 'darkgreen'
behave_col = 'darkblue'
study_col = 'purple'

### Load data
tresids = pd.read_stata("temp/teach_mean_resids.dta").drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behave_r', axis=1).values[:,:]

college = tresids.filter(regex='college_bound_', axis=1).values[:,:]

crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]


#####################################
### 1) Crime/college attendence
#####################################

### i) Crime/college value added is observed (oracle)
def oracal_fun(origX, origY, bottom_Xpercent = 0.05):

    '''
    IMPORTANT: origX and origY need to be adjusted such that positive values are GOOD (e.g, crime becomes -crime)  
    '''

    # Variance-covariance matrix (Sigma) of origX and origY 
    varx = ustat.varcovar(origX, origX)
    vary = ustat.varcovar(origY, origY)
    covar = ustat.varcovar(origY, origX)
    

    # Means (over teachers)
    mux = np.nanmean(np.nanmean(origX, axis=1))
    muy = np.nanmean(np.nanmean(origY, axis=1))

    # Loop over grid of weights
    changeX = []
    changeY = []
    for w in np.linspace(0,1, 100):
        var_index = np.power(w, 2)*varx + np.power(1-w, 2)*vary + 2*w*(1-w)*covar# var index
        mu_index = w*mux + (1-w)*muy # mean index
        covar_withX = w*varx + (1-w)*covar
        covar_withY = w*covar + (1-w)*vary

        # Fire teachers with index below q
        q = norm.ppf(bottom_Xpercent, loc = mu_index, scale = np.power(var_index, 0.5)) # bottom X% according to index  

        # Change in X
        rho = covar_withX/np.power(var_index*varx, 0.5)
        meanBottom = mux + rho*np.power(varx/var_index, 0.5) * -np.power(var_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_index, 0.5)))
        changeX += [mux - meanBottom, ]

        rho = covar_withY/np.power(var_index*vary, 0.5)
        meanBottom = muy + rho*np.power(vary/var_index, 0.5) * -np.power(var_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_index, 0.5)))
        changeY += [muy - meanBottom, ]
        del rho, meanBottom

    return np.array(changeX), np.array(changeY)
    


def shortrunOrecal_fun(origX, origY, cog = cog, behave = behave, study = study, bottom_Xpercent = 0.05, grid_len=10):

    # Variance-covariance matrix (Sigma) of origX and origY 
    varx = ustat.varcovar(origX, origX)
    vary = ustat.varcovar(origY, origY)

    # Means (over teachers)
    mux = np.nanmean(np.nanmean(origX, axis=1))
    muy = np.nanmean(np.nanmean(origY, axis=1))
    
    # Short-run covariance with long-run
    covarXcog = ustat.varcovar(origX, cog)
    covarXbehave = ustat.varcovar(origX, behave)
    covarXstudy = ustat.varcovar(origX, study)

    covarYcog = ustat.varcovar(origY, cog)
    covarYbehave = ustat.varcovar(origY, behave)
    covarYstudy = ustat.varcovar(origY, study)

    # Short-run var-covar
    varCog = ustat.varcovar(cog, cog)
    varBehave = ustat.varcovar(behave, behave)
    varStudy = ustat.varcovar(study, study)

    covar_cog_behave = ustat.varcovar(behave, cog)
    covar_cog_study = ustat.varcovar(study, cog)
    covar_study_behave = ustat.varcovar(study, behave)

    # Short-run means (over teachers)
    mu_cog = np.nanmean(np.nanmean(cog, axis=1))
    mu_behave = np.nanmean(np.nanmean(behave, axis=1))
    mu_study = np.nanmean(np.nanmean(study, axis=1))


    # Three dimensional grid
    weights = []
    for w1 in range(grid_len+1):
        for w2 in range(grid_len-w1+1):
            # print('W1: {} and W2 {}'.format(w1, w2))
            _w1 = w1/grid_len
            _w2 = w2/grid_len
            weights += [(_w1,_w2,1-_w1-_w2),]
            del _w1, _w2

    # Loop over grid of weights
    changeX = []
    changeY = []
    lsw1 = []
    lsw2 = []
    lsw3 = []
    for w1, w2, w3 in tqdm(weights):

        var_index = np.power(w1, 2)*varCog + np.power(w2, 2)*varBehave + np.power(w3, 2)*varStudy + 2*w1*w2*covar_cog_behave + 2*w1*w3*covar_cog_study + 2*w2*w3*covar_study_behave # var index
        mu_index = w1*mu_cog + w2*mu_behave + w3*mu_study # mean index
        covar_withX = w1*covarXcog + w2*covarXbehave + w3*covarXstudy
        covar_withY = w1*covarYcog + w2*covarYbehave + w3*covarYstudy

        # Fire teachers with index below q
        q = norm.ppf(bottom_Xpercent, loc = mu_index, scale = np.power(var_index, 0.5)) # bottom X% according to index  

        # Change in X
        rho = covar_withX/np.power(var_index*varx, 0.5)
        meanBottom = mux + rho*np.power(varx/var_index, 0.5) * -np.power(var_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_index, 0.5)))
        changeX += [mux - meanBottom, ]

        rho = covar_withY/np.power(var_index*vary, 0.5)
        meanBottom = muy + rho*np.power(vary/var_index, 0.5) * -np.power(var_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_index, 0.5)))
        changeY += [muy - meanBottom, ]
        del rho, meanBottom

        lsw1 += [w1,]
        lsw2 += [w2,]
        lsw3 += [w3,]

    return np.array(changeX), np.array(changeY), np.array(lsw1), np.array(lsw2), np.array(lsw3)


def shortrunEB_fun(origX, origY, cog = cog, behave = behave, study = study, bottom_Xpercent = 0.05, grid_len=10):

    # TRUE variance-covariance matrix of origX and origY 
    varx = ustat.varcovar(origX, origX)
    vary = ustat.varcovar(origY, origY)

    # Means (over teachers)
    mux = np.nanmean(np.nanmean(origX, axis=1))
    muy = np.nanmean(np.nanmean(origY, axis=1))
    
    # Covariance of TRUE teacher effects on short-run with long-run
    covarXcog = ustat.varcovar(origX, cog)
    covarXbehave = ustat.varcovar(origX, behave)
    covarXstudy = ustat.varcovar(origX, study)

    covarYcog = ustat.varcovar(origY, cog)
    covarYbehave = ustat.varcovar(origY, behave)
    covarYstudy = ustat.varcovar(origY, study)

    # Short-run TRUE teacher effects var-covar
    varCog = ustat.varcovar(cog, cog)
    varBehave = ustat.varcovar(behave, behave)
    varStudy = ustat.varcovar(study, study)

    covar_cog_behave = ustat.varcovar(behave, cog)
    covar_cog_study = ustat.varcovar(study, cog)
    covar_study_behave = ustat.varcovar(study, behave)

    # Short-run OBSERVED teacher effects var-covar
    varCog_hat = np.nanvar(cog.ravel())
    varBehave_hat = np.nanvar(behave.ravel())
    varStudy_hat = np.nanvar(study.ravel())

    covar_cog_behave_hat = pd.DataFrame({'a':behave.ravel(), 'b':cog.ravel()}).cov().values[0,1]  
    covar_cog_study_hat = pd.DataFrame({'a':study.ravel(), 'b':cog.ravel()}).cov().values[0,1]
    covar_study_behave_hat = pd.DataFrame({'a':behave.ravel(), 'b':study.ravel()}).cov().values[0,1]

    # Short-run means (over teachers)
    mu_cog = np.nanmean(np.nanmean(cog, axis=1))
    mu_behave = np.nanmean(np.nanmean(behave, axis=1))
    mu_study = np.nanmean(np.nanmean(study, axis=1))

    # Projection coefficients of X (or Y) on cog, behave, and study skills
    Sigma_mumu = np.array([
        [varCog_hat, covar_cog_behave_hat, covar_cog_study_hat],
        [covar_cog_behave_hat, varBehave_hat, covar_study_behave_hat],
        [covar_cog_study_hat, covar_study_behave_hat, varStudy_hat]
        ])

    Sigma_muY = np.array([covarYcog, covarYbehave, covarYstudy])
    Sigma_muX = np.array([covarXcog, covarXbehave, covarXstudy])

    bX = np.linalg.inv(Sigma_mumu).dot(Sigma_muX)
    bY = np.linalg.inv(Sigma_mumu).dot(Sigma_muY)

    # Variance of EB X and EB Y (note, mean of EB X and EB Y are the same as those of X and Y)
    var_ebX = np.power(bX[0], 2)*varCog_hat + np.power(bX[1], 2)*varBehave_hat + np.power(bX[2], 2)*varStudy_hat + 2*bX[0]*bX[1]*covar_cog_behave_hat + 2*bX[0]*bX[2]*covar_cog_study_hat + 2*bX[1]*bX[2]*covar_study_behave_hat
    var_ebY = np.power(bY[0], 2)*varCog_hat + np.power(bY[1], 2)*varBehave_hat + np.power(bY[2], 2)*varStudy_hat + 2*bY[0]*bY[1]*covar_cog_behave_hat + 2*bY[0]*bY[2]*covar_cog_study_hat + 2*bY[1]*bY[2]*covar_study_behave_hat

    # Three dimensional grid
    weights = []
    for w1 in range(grid_len+1):
        for w2 in range(grid_len-w1+1):
            # print('W1: {} and W2 {}'.format(w1, w2))
            _w1 = w1/grid_len
            _w2 = w2/grid_len
            weights += [(_w1,_w2,1-_w1-_w2),]
            del _w1, _w2

    # Loop over grid of weights
    changeX = []
    changeY = []
    lsw1 = []
    lsw2 = []
    lsw3 = []
    for w1, w2, w3 in tqdm(weights):

        # EB of index using cog, behave, and study (i.e., project TRUE index on cog, behave, and study)
        Sigma_muIndex = np.array([
            w1*(varCog + covar_cog_behave + covar_cog_study),
            w2*(varBehave + covar_cog_behave + covar_study_behave),
            w3*(varStudy + covar_cog_study + covar_study_behave)])

        # Projection coefficients of Index on cog, behave, and study
        bI = np.linalg.inv(Sigma_mumu).dot(Sigma_muIndex)

        # EB index expectation and variance        
        mu_index = w1*mu_cog + w2*mu_behave + w3*mu_study # mean index (Note, E[ideal index] = E[EB index])
        var_EB_index = np.power(bI[0], 2)*varCog_hat + np.power(bI[1], 2)*varBehave_hat + np.power(bI[2], 2)*varStudy_hat + 2*bI[0]*bI[1]*covar_cog_behave_hat + 2*bI[0]*bI[2]*covar_cog_study_hat + 2*bI[1]*bI[2]*covar_study_behave_hat # var index

        # Fire teachers with index below q
        q = norm.ppf(bottom_Xpercent, loc = mu_index, scale = np.power(var_EB_index, 0.5)) # bottom X% according to index  

        # Covariance of EB index with long-run outcomes
        covarX_EBindex =  bX[0] * (bI[0]*varCog_hat + bI[1]*covar_cog_behave_hat + bI[2]*covar_cog_study_hat) + bX[1] * (bI[0]*covar_cog_behave_hat + bI[1]*varBehave_hat + bI[2]*covar_study_behave_hat) +  bX[2] * (bI[0]*covar_cog_study_hat + bI[1]*covar_study_behave_hat + bI[2]*varStudy_hat)
        covarY_EBindex =  bY[0] * (bI[0]*varCog_hat + bI[1]*covar_cog_behave_hat + bI[2]*covar_cog_study_hat) + bY[1] * (bI[0]*covar_cog_behave_hat + bI[1]*varBehave_hat + bI[2]*covar_study_behave_hat) +  bY[2] * (bI[0]*covar_cog_study_hat + bI[1]*covar_study_behave_hat + bI[2]*varStudy_hat)

        # Change in X
        rho = covarX_EBindex/np.power(var_EB_index*var_ebX, 0.5)
        meanBottom = mux + rho*np.power(var_ebX/var_EB_index, 0.5) * -np.power(var_EB_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_EB_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_EB_index, 0.5)))
        changeX += [mux - meanBottom, ]

        # Change in Y
        rho = covarY_EBindex/np.power(var_EB_index*var_ebY, 0.5)
        meanBottom = muy + rho*np.power(var_ebY/var_EB_index, 0.5) * -np.power(var_EB_index, 0.5) * (norm.pdf((q-mu_index)/np.power(var_EB_index, 0.5))/norm.cdf((q-mu_index)/np.power(var_EB_index, 0.5)))
        changeY += [muy - meanBottom, ]
        del rho, meanBottom

        lsw1 += [w1,]
        lsw2 += [w2,]
        lsw3 += [w3,]

    return np.array(changeX), np.array(changeY), np.array(lsw1), np.array(lsw2), np.array(lsw3)



### College and criminal arrest
oracalCrime, oracalCollege = oracal_fun(-crime, college)
sr_oracalCrime, sr_oracalCollege, w1, w2, w3 = shortrunOrecal_fun(-crime, college, grid_len=50)
sr = pd.DataFrame({'college':sr_oracalCollege, 'crime':sr_oracalCrime, 'w1':w1, 'w2':w2})


# Full weight points
full_weight = sr.loc[(sr.w1==1) | (sr.w2==1) | ((sr.w1==0) & (sr.w2==0))].copy()
full_weight.reset_index(inplace=True, drop=True)
full_weight.sort_values(['w1','w2'], ascending=[True, True], inplace=True)

# Frontier
sr = sr.loc[(sr[['college','crime']].values[:,None] <= sr[['college','crime']].values).all(axis=2).sum(axis=1) == 1].reset_index(drop=True)

# # EB
sr_ebCrime, sr_ebCollege, w1, w2, w3 = shortrunEB_fun(-crime, college, grid_len=50)
srEB = pd.DataFrame({'college':sr_ebCollege, 'crime':sr_ebCrime, 'w1':w1, 'w2':w2})

# # Full weight points
full_weight_eb = srEB.loc[(srEB.w1==1) | (srEB.w2==1) | ((srEB.w1==0) & (srEB.w2==0))].copy()
full_weight_eb.reset_index(inplace=True, drop=True)

srEB = srEB.loc[(srEB[['college','crime']].values[:,None] <= srEB[['college','crime']].values).all(axis=2).sum(axis=1) == 1].reset_index(drop=True)


# Figure
fig, axes = plt.subplots(figsize=(8,5))
axes.plot(oracalCollege, oracalCrime, linestyle = lstl_oracal, color = col_oracal, label = 'True effects on arrest and college')

# Oracal short-run
axes.plot(sr['college'], sr['crime'], linestyle = lstl_SRoracal, color = col_SRoracal, label = 'True effects on short-run outcomes')
axes.scatter(full_weight.iloc[0,:]['college'], full_weight.iloc[0,:]['crime'], color= study_col, marker= study_marker)
# label = 'Only study skills (shaded w/ EB)'
axes.annotate('Only study skills', (full_weight.iloc[0,:]['college']+0.001, full_weight.iloc[0,:]['crime']-0.001), color = study_col)

#; 
axes.scatter(full_weight.iloc[1,:]['college'], full_weight.iloc[1,:]['crime'], color = behave_col , marker= behave_marker)
# label = 'Only behavioral (shaded w/ EB)'
axes.annotate('Only behavioral', (full_weight.iloc[1,:]['college']+0.002, full_weight.iloc[1,:]['crime']), color = behave_col, alpha=1)


axes.scatter(full_weight.iloc[2,:]['college'], full_weight.iloc[2,:]['crime'], color = cog_col, marker= cog_marker)
# label = 'Only test scores (shaded w/ EB)'
axes.annotate('Only test scores', (full_weight.iloc[2,:]['college']+0.001, full_weight.iloc[2,:]['crime']), color = cog_col, alpha=1)

# EB short-run
axes.plot(srEB['college'], srEB['crime'], linestyle = lstl_ebSRoracal, color = col_ebSRoracal, label = 'EB posteriors of effects on short-run outcomes')

axes.scatter(full_weight_eb.iloc[0,:]['college'], full_weight_eb.iloc[0,:]['crime'], color='none', edgecolors = study_col, marker= study_marker)
# axes.annotate('Only study skills', (full_weight_eb.iloc[0,:]['college']+0.001, full_weight_eb.iloc[0,:]['crime']-0.001), color = study_col, alpha=study_alpha)

axes.scatter(full_weight_eb.iloc[1,:]['college'], full_weight_eb.iloc[1,:]['crime'], color='none', edgecolors = behave_col , marker= behave_marker)
# axes.annotate('Only Behavioral', (full_weight_eb.iloc[1,:]['college']+0.001, full_weight_eb.iloc[1,:]['crime']-0.001), color = behave_col, alpha=behave_alpha)


axes.scatter(full_weight_eb.iloc[2,:]['college'], full_weight_eb.iloc[2,:]['crime'], color='none', edgecolors = cog_col, marker= cog_marker)
# axes.annotate('Only Test scores', (full_weight_eb.iloc[2,:]['college']+0.001, full_weight_eb.iloc[2,:]['crime']+0.001), color = cog_col, alpha=cog_alpha)

axes.scatter(full_weight_eb.iloc[2,:]['college'], full_weight_eb.iloc[2,:]['crime'], color='none', marker= '.', label = '\nMarkers indicate putting full weight\non effects on a single short-run outcome')


axes.set_ylabel('Reduction in future criminal arrest \n in exposed classrooms') 
axes.set_xlabel('Increase in college attendance \n in exposed classrooms') 
axes.set_xlim(np.min([full_weight_eb['college'].min(),full_weight['college'].min()-0.005,0, oracalCollege.min()]), oracalCollege.max()+0.005)
axes.legend(fancybox=True, edgecolor="black")
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figure5.pdf')

