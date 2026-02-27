#teacher ustat
import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys
from multiprocessing import Pool

# Ustat functions
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
# Color options
col_grad = 'blue'
col_gpa = 'royalblue'
col_college = 'purple'

col_aoc_traff = 'gold'
col_crimeany = 'orange'
col_crime = 'peru'
col_aoc_index = 'red'
col_aoc_incar = 'darkred'

hatch1 = ''
hatch2 = '/'
hatch3 = 'X'
hatch4 = '+'


##########################################################################
###  Variance of teacher effects across groups
##########################################################################

# Graph of correlation in effects
covs = [('white','Race (white vs. non-white)'),('male','Gender'),('disadv','Econ. disadvantaged'),('atrisk','Arrest risk')]
results = {}
for covariate, covname in covs:
    results[covname] = {}
    tresids = pd.read_stata("temp/teach_mean_resids_cov{}.dta").drop('teachid', axis=1)
    cog0 = tresids.filter(regex='testscores_r0', axis=1).values[:,:]
    cog1 = tresids.filter(regex='testscores_r1', axis=1).values[:,:]
    behave0 = tresids.filter(regex='behavpca_r0', axis=1).values[:,:]
    behave1 = tresids.filter(regex='behavpca_r1', axis=1).values[:,:]
    study0 = tresids.filter(regex='studypca_r0', axis=1).values[:,:]
    study1 = tresids.filter(regex='studypca_r1', axis=1).values[:,:]
    anycrime0 = tresids.filter(regex='aoc_any_r0', axis=1).values[:,:]
    anycrime1 = tresids.filter(regex='aoc_any_r1', axis=1).values[:,:]   
    crime0 = tresids.filter(regex='aoc_crim_r0', axis=1).values[:,:]
    crime1 = tresids.filter(regex='aoc_crim_r1', axis=1).values[:,:]
    incar0 = tresids.filter(regex='aoc_incar_r0', axis=1).values[:,:]
    incar1 = tresids.filter(regex='aoc_incar_r1', axis=1).values[:,:]
    index0 = tresids.filter(regex='aoc_index_r0', axis=1).values[:,:]
    index1 = tresids.filter(regex='aoc_index_r1', axis=1).values[:,:]
    gpa0 = tresids.filter(regex='gpa_weighted_r0', axis=1).values[:,:]
    gpa1 = tresids.filter(regex='gpa_weighted_r1', axis=1).values[:,:]
    grad0 = tresids.filter(regex='grad_r0', axis=1).values[:,:]
    grad1 = tresids.filter(regex='grad_r1', axis=1).values[:,:]
    college0 = tresids.filter(regex='college_bound_r0', axis=1).values[:,:]
    college1 = tresids.filter(regex='college_bound_r1', axis=1).values[:,:]
    for out, d0, d1 in [('Test scores',cog0,cog1),('Study skills',study0,study1),('Behaviors',behave0,behave1),
                    ('Any CJC',anycrime0,anycrime1), ('Criminal arrest',crime0,crime1),('Index crime',index0,index1),('Incarceration',incar0,incar1), 
                    ('12th grade GPA',gpa0,gpa1), ('Graduation',grad0,grad1), ('College attendance',college0,college1)]:
        print('\n Working on {} and {}: outcome = {} \n'.format(covariate, covname, out))

        # SD ests
        results[covname][out+'1'] = ustat.sd_func(d1)
        results[covname][out+'1_se'] = ustat.sd_samp_var(d1)**0.5

        results[covname][out+'0'] =  ustat.sd_func(d0)
        results[covname][out+'0_se'] = ustat.sd_samp_var(d0)**0.5


tab = pd.DataFrame({'statistic':np.nan, 'subgroup':' ', 'outcome':np.nan, 'value':np.nan}, index = [0])
ls = []
for covariate, covname in covs:
    for var in ['Test scores','Study skills','Behaviors','Any CJC','Criminal arrest','Index crime','Incarceration','12th grade GPA','Graduation','College attendance']:
        for l in [0,1]:
            tab['subgroup'] = covariate + '{}'.format(l)
            tab['statistic'] = 'SD'
            tab['outcome'] = var
            tab['value'] = results[covname][var + '{}'.format(l)]
            ls += [tab.copy(),]

            tab['statistic'] = 'SDse'
            tab['value'] = results[covname][var + '{}_se'.format(l)]
            ls += [tab.copy(),]
tab = pd.concat(ls)
tab.reset_index(inplace=True, drop = True)
tab.to_stata('temp/hetero_SDs.dta')
