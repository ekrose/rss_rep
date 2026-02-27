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

hatch_crimeany = ''
hatch_crime = 'X'
hatch_index = '/'
hatch_incar = '+'

hatch_gpa = '/'
hatch_grad = 'X'
hatch_college = ''


####################################
### Graphs of 1 SD effects
####################################

means = {'Any CJC':0.44, 'Criminal arrest':0.24, 'Index crime':0.1, 'Incarceration': 0.089, '12th grade GPA':3.13, 'College attendance':0.46,  'Graduation':0.91}

# Load data
tresids = pd.read_stata("temp/teach_mean_resids.dta".format(
        spec, droppval)).drop('teachid', axis=1)
cog = tresids.filter(regex='^lead1_testscores_', axis=1).values[:,:]
study = tresids.filter(regex='^lead1_studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behave_r', axis=1).values[:,:]

gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
aoc_traff = tresids.filter(regex='aoc_traff_r', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]


labels = ['Test scores', 'Behaviors', 'Study skills']

effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index, 'Incarceration':aoc_incar, '12th grade GPA':gpa, 'College attendance':college, 'Graduation':grad}
results = pd.DataFrame(columns=effects.keys(), index=labels).drop(labels, axis=1)
for row in results.index:
    for idx, col in enumerate(results.columns):
        print('Working on effect of {} on {}'.format(row, col))
        if row != col:
            effect = ustat.sd_effect_func(effects[row], effects[col])
            se = ustat.sd_effect_samp_covar(effects[row], effects[col])**0.5
            results.loc[row, col] = (effect, se)

# Only academic long-run
fig, axes = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.2
for adj, lab, cl, hatch in [(-1.5, '12th grade GPA', col_gpa, hatch_gpa), (-0.5, 'Graduation', col_grad, hatch_grad), (0.5, 'College attendance', col_college, hatch_college)]:
    axes.bar(x+adj*width, results.loc[labels, lab].apply(lambda x: x[0]).values,
            width,
            yerr=results.loc[labels, lab].apply(lambda x: x[1]).values*1.96,
            label=lab,
            color=cl,
            hatch=hatch)
    for cc,l in enumerate(labels):
        offset_dirc = results.loc[l, lab][0]
        axes.annotate('{:4.1f}%'.format( results.loc[l, lab][0]/means[lab] * 100 ),xy=(x[cc]+adj*width, results.loc[l, lab][0]),
            xytext=(-20 , (offset_dirc < 0) * -20 + (offset_dirc > 0) * 10), # X points vertical offset
            textcoords="offset points",
            va='bottom')

axes.set_ylabel('Effect of 1 SD increase on long-run outcomes') 
axes.set_xlabel('Short-run outcome') 
axes.set_xticks(x)
axes.set_xticklabels(labels)
axes.legend()
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figureA2b.pdf')

# Only CJC long-run 
fig, axes = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.2
for adj, lab, cl, hatch in [(-1.5, 'Any CJC', col_crimeany, hatch_crimeany), (-0.5, 'Criminal arrest', col_crime, hatch_crime), (0.5, 'Index crime', col_aoc_index, hatch_index), (1.5, 'Incarceration', col_aoc_incar, hatch_incar)]:
    axes.bar(x+adj*width, results.loc[labels, lab].apply(lambda x: x[0]).values,
            width,
            yerr=results.loc[labels, lab].apply(lambda x: x[1]).values*1.96,
            label=lab,
            color=cl,
            hatch=hatch)
    for cc,l in enumerate(labels):
        offset_dirc = results.loc[l, lab][0]
        axes.annotate('{:4.1f}%'.format( results.loc[l, lab][0]/means[lab] * 100 ),xy=(x[cc]+adj*width, results.loc[l, lab][0]),
            xytext=(-20 , (offset_dirc < 0) * -20 + (offset_dirc > 0) * 10), # X points vertical offset
            textcoords="offset points",
            va='bottom')


axes.set_ylabel('Effect of 1 SD increase on long-run outcomes') 
axes.set_xlabel('Short-run outcome') 
axes.set_xticks(x)
axes.set_xticklabels(labels)
axes.legend()
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figureA2a.pdf')




