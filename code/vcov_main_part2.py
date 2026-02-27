import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys
from multiprocessing import Pool

# import ustat functions
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


################################################
### Covariance of short and long-run outcomes
################################################

# Load the data
tresids = pd.read_stata("temp/teach_mean_resids.dta".format(
        spec, droppval)).drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behave_r', axis=1).values[:,:]

gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
aoc_traff = tresids.filter(regex='aoc_traff_r', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]

### Covariance with short-run outcomes
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
results_raw_est = pd.DataFrame(columns=effects.keys(), index=effects.keys())
results_raw_estSE = pd.DataFrame(columns=effects.keys(), index=effects.keys())

# First populate SDs
for idx, row in enumerate(results.columns):
    print('Working on SD of outcome: {}'.format(row), flush=True)
    sdev = ustat.sd_func(effects[row])
    se = ustat.sd_samp_var(effects[row])**0.5
    results.loc[row, row] = "{:4.3f} ({:5.4f})".format(sdev, se)
    results_raw_est.loc[row, row] = sdev
    results_raw_estSE.loc[row, row] = se

# Now add correlations
for idx, row in enumerate(results.columns):
    for col in results.iloc[idx+1:].index:
        print('Working on CORR of outcomes: {} and {}'.format(row, col), flush=True)
        sdev = ustat.correl_func(effects[row], effects[col])
        se = ustat.corr_samp_covar(effects[row], effects[col])**0.5
        results.loc[row, col] = "{:4.3f} ({:4.3f})".format(sdev, se)
        results_raw_est.loc[row, col] = sdev
        results_raw_estSE.loc[row, col] = se

### Save to latex
_ls = ['Any CJC', 'Criminal arrest', 'Index crime', 'Incarceration', '12th grade GPA','College attendance','Graduation']
tmp = results.filter(items = _ls, axis=1).filter(items = _ls, axis=0)
tmp.to_latex('tables/table2.tex' 
    na_rep='', escape=False, multicolumn_format = 'c', column_format = 'c' * int(tmp.shape[1] + 1)
    ) 




