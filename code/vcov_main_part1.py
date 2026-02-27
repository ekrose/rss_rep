import pandas as pd
import numpy as np
from scipy import sparse
from tqdm import tqdm
import os, sys
from multiprocessing import Pool
np.random.seed(93293483)

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


#####################################
### 1) Short-run outcomes
#####################################

# Load the data
tresids = pd.read_stata("temp/teach_mean_resids.dta").drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behave_r', axis=1).values[:,:]
oss = tresids.filter(regex='^lead1_oss_r', axis=1).values[:,:]

effects = {'Test scores':cog, 'Math scores':math, 'Reading scores':eng, 'Study skills':study, 'Behaviors':behave}
results = pd.DataFrame(columns=effects.keys(), index=effects.keys())

## Main var covar
# First populate SDs
for idx, row in enumerate(results.columns):
    sdev = ustat.sd_func(effects[row])
    se = ustat.sd_samp_var(effects[row])**0.5
    results.loc[row, row] = "{:4.3f} ({:5.4f})".format(sdev, se)
    print("{:4.3f} ({:5.4f})".format(sdev, se), flush=True)

# Now add correlations
for idx, row in enumerate(results.columns):
    for col in results.iloc[idx+1:].index:
        sdev = ustat.correl_func(effects[row], effects[col])
        se = ustat.corr_samp_covar(effects[row], effects[col])**0.5
        results.loc[row, col] = "{:4.3f} ({:5.4f})".format(sdev, se)
        print("{:4.3f} ({:5.4f})".format(sdev, se), flush=True)

# Save to latex
results.to_latex('tables/table3.tex', 
    na_rep='', escape=False, 
    multicolumn_format = 'c',
    column_format = 'c' * int(results.shape[1] + 1)
    )  

# Compare to other estimates in tthe literature
fig, axes = plt.subplots(figsize=(8,5))
estimates = [(ustat.sd_func(effects['Math scores']),
                ustat.sd_func(effects['Reading scores']), 'this paper'),
            (0.163,0.124, 'CFR 2014 (elem)'),
            (0.134,0.098, 'CFR 2014 (mid)'),
            (0.0751,0.0204, 'Jackson 2018 (9th g)'),
            (0.228,0.189, 'Bacher-Hicks e.a. 2014 (elem)'),
            (0.206,0.097, 'Bacher-Hicks e.a. 2014 (mid)'),
            (0.193,0.150, 'Kane Staiger 2008 (elem)'),
            (0.150,0.109, 'Rothstein 2010 (5th g)'),
            (0.142,0.126, 'Rothstein 2010 (4th g)')
                ]    # math, eng, paper

axes.scatter(x=[a[0] for a in estimates], y=[a[1] for a in estimates], color=['red'] + ['blue' for b in range(len(estimates)-1)])
for x,y,text in estimates:
    if text == 'this paper':
        col = 'darkred'
    else:
        col = 'darkblue'
    if text == 'Rothstein 2010 (4th g)':
        axes.annotate(text, (x-.075,y+.002), color=col, size=9)    
    else:
        axes.annotate(text, (x+.002,y+.002), color=col, size=9)
axes.set_xlabel('SD of math effects') 
axes.set_ylabel('SD of reading effects') 
axes.set_xlim(0,0.35)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figureA1.pdf')



