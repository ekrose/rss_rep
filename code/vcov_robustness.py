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

import glob
import re

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
### 1) Estimation
##################################### 

if True:
    # Fix the format from robust_options.txt
    lines = robust_options = pd.read_csv('code/robust_options.txt').iloc[:,0].tolist()
    ls = []
    for cc, l in enumerate(lines):
        if cc == 0:
            ls.append(l)
        else:
            _t = l.startswith('> ')
            if _t:
                ls[-1] = ls[-1]+l.replace('> ', '')
            else:
                ls.append(l)

    # check which specification have already been estimated and remove them
    done_specs = glob.glob("temp/robust/resids_iter*")
    done_iters = [int(re.findall(r'[0-9]+', l)[0])  for l in done_specs]

    # Send code to cluster
    for cc in range(len(ls)):
        if cc in done_iters:
            print('Option number already estimated: {} \n'.format(cc)) 
            pass
        else:
            opt = ls[cc]
            print('Working on option number: {} \n'.format(cc)) 
            print('and specification: {} \n \n \n \n \n'.format(opt)) 
            os.system('bash script_robust_options.sh {} \"{}\"'.format(cc, opt))


# Plot sensitivities
specs = glob.glob("temp/robust/*.dta")

allresults = []
for idx, spec in enumerate(specs):
    print("Working on {} out of {}".format(idx,len(specs)))
    tresids = pd.read_stata(spec).drop('teachid', axis=1)
    cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
    behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]
    crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]

    try:
        ### 1 SD effects
        short_run = {'Test scores':cog, 'Behaviors':behave}
        long_run = {'Criminal arrest':crime}
        results = pd.DataFrame(columns=long_run.keys(), index=short_run.keys())
        for row in results.index:
            for idx, col in enumerate(results.columns):
                if row != col:
                    effect = ustat.sd_effect_func(short_run[row], long_run[col])
                    results.loc[row, col] = effect
        results['spec'] = spec
        allresults += [results.reset_index(),]
    except:
        print("Couldn't estimate spec: {}".format(spec))

results = pd.concat(allresults, ignore_index=True).rename(columns={'index':'short_run'})

# Read lines
lines = []
row = 1
with open("code/robust_options.txt") as opts:
    for line in opts:
        if line[0] == ">":
            lines[-1][1] = lines[-1][1] + line[2:].strip()
        else:
            lines += [[str(row), line.strip()]]
            row += 1
specs = pd.DataFrame(lines, columns=['iter','sec'])

# Add to results
results['iter'] = results.spec.str.extract("iter([0-9]+).dta")
results = results.merge(specs, how='left', on='iter')
def fun_numVars(x):
    try:
        return len(x.split(" "))
    except:
        print('Error in numVars (length) calculation')
        return np.nan

results['nterms'] = results.sec.apply(lambda x: fun_numVars(x))

results['crim'] = pd.to_numeric(results['Criminal arrest'])
csent = results.groupby(['nterms','short_run']).crim.agg(['mean','median','min','max']).unstack()

# Main effects
tresids = pd.read_stata("temp/teach_mean_resids.dta").drop('teachid', axis=1)
cog = tresids.filter(regex='^testscores_', axis=1).values[:,:]
math = tresids.filter(regex='^math_', axis=1).values[:,:]
eng = tresids.filter(regex='^eng_', axis=1).values[:,:]
study = tresids.filter(regex='^studypca_', axis=1).values[:,:]
behave = tresids.filter(regex='^behavpca_', axis=1).values[:,:]

gpa = tresids.filter(regex='gpa_weighted_', axis=1).values[:,:]
college = tresids.filter(regex='college_bound_', axis=1).values[:,:]
grad = tresids.filter(regex='^grad_', axis=1).values[:,:]

crime = tresids.filter(regex='aoc_crim_r', axis=1).values[:,:]
crimeany = tresids.filter(regex='aoc_any_r', axis=1).values[:,:]
aoc_traff = tresids.filter(regex='aoc_traff_r', axis=1).values[:,:]
aoc_index = tresids.filter(regex='aoc_index_r', axis=1).values[:,:]
aoc_incar = tresids.filter(regex='aoc_incar_r', axis=1).values[:,:]

### Covariance with short-run outcomes
effects = {'Test scores':cog, 'Behaviors':behave, 'Study skills':study, 'Criminal arrest':crime, 'Incarceration':aoc_incar, 'College attendance':college, 'Graduation':grad, }
main_results = pd.DataFrame(columns=effects.keys(), index=effects.keys())
for row in main_results.index:
    for idx, col in enumerate(main_results.columns):
        if row != col:
            _effect = ustat.sd_effect_func(effects[row], effects[col])
            main_results.loc[row, col] = _effect

main_covdesign = 'i.year##i.grade##i.subj i.grade##i.subj##c.lag1_math* i.grade##i.subj##c.lag1_read* i.grade##c.sgyts_mean_* i.grade##c.s_mean_* exc_not exc_aig exc_behav exc_educ aigmath aigread disadv lim_eng female black white grade_rep lag1_daysabs lag1_any_discp'
main_numVars = len(main_covdesign.split(" "))

# Test scores
fig, axes = plt.subplots(figsize=(8,5))
axes.scatter(x=csent.index, y=csent[('median','Test scores')], label='Median effect')
axes.plot(csent.index, csent[('min','Test scores')], 'r--x', label='Min')
axes.plot(csent.index, csent[('max','Test scores')], 'r--s', label='Max')
axes.hlines(y=main_results.loc['Test scores','Criminal arrest'], xmin=5, xmax=26,linestyle='--', alpha=0.8, color = 'green')
axes.set_xlabel('# of controls included in model')
axes.set_ylabel('Impact of 1 SD increase in\ntest score effects on arrests') 
axes.set_ylim(-0.015,0.005)
axes.annotate("Preferred estimate",
            xy=(main_numVars, main_results.loc['Test scores','Criminal arrest']), xycoords='data',
            xytext=(11.5, -0.006), textcoords='data',
            size=11, va="center", ha="center", color = 'green',
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2", color = 'green'),
            )
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=3)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figureA4a.pdf')

# Beahvioral
fig, axes = plt.subplots(figsize=(8,5))
axes.scatter(x=csent.index, y=csent[('median','Behaviors')], label='Median effect', color='b')
axes.plot(csent.index, csent[('min','Behaviors')], 'b--x', label='Min')
axes.plot(csent.index, csent[('max','Behaviors')], 'b--s', label='Max')
axes.hlines(y=main_results.loc['Behaviors','Criminal arrest'], xmin=5, xmax=26, linestyle='--', alpha=0.8, color = 'green')
axes.set_xlabel('# of controls included in model')
axes.set_ylabel('Impact of 1 SD increase in\nbehavioral effects on arrests') 
axes.set_ylim(-0.015,0.005)
axes.annotate("Preferred estimate",
            xy=(main_numVars, main_results.loc['Behaviors','Criminal arrest']), xycoords='data',
            xytext=(11.5, 0.001), textcoords='data',
            size=11, va="center", ha="center", color = 'green',
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2", color = 'green'),
            )
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=3)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figureA4b.pdf')

