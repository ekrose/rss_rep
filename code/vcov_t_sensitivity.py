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

# Helpers
def weighted_nanmean(values, weights):
    values = np.array(values)
    weights = np.array(weights)

    # Mask NaNs
    mask = ~np.isnan(values)
    
    return np.average(values[mask], weights=weights[mask])


### 1) Load the long data
tresids = pd.read_stata("temp/teach_mean_resids_long.dta")

# Go wide, including year vars
r_cols = ['year'] + [col for col in tresids.columns if col.endswith("_r")]
resids_wide = tresids.pivot(index='teachid', columns='obs', values=r_cols)
resids_wide.columns = [f"{var}_{int(obs)}" for var, obs in resids_wide.columns]
resids_wide = resids_wide.reset_index()

# Get outcomes
years = resids_wide.filter(regex='^year_', axis=1).values[:,:]
cog = resids_wide.filter(regex='^testscores_', axis=1).values[:,:]
behave = resids_wide.filter(regex='^behave_', axis=1).values[:,:]
study = resids_wide.filter(regex='^studypca_', axis=1).values[:,:]
crimeany = resids_wide.filter(regex='aoc_any_', axis=1).values[:,:]
crime = resids_wide.filter(regex='aoc_crim_r', axis=1).values[:,:]
aoc_index = resids_wide.filter(regex='aoc_index_r', axis=1).values[:,:]

effects = {'Test scores':cog, 'Study skills':study, 'Behaviors':behave,
    'Any CJC':crimeany, 'Criminal arrest':crime, 'Index crime':aoc_index,}
results = []

### 2) Compute sensitivity tests
# Populated
for effect in effects.keys():
    for mint in range(1,4):
        for maxt in list(range(mint,4)) + [None]:

            sd = np.power(ustat.varcovar_gaps(
                    effects[effect], effects[effect], 
                        years, mint, maxt), 0.5) 
            t_corr = ustat.correl_func_gaps(
                    effects[effect], cog,
                        years, mint, maxt)
            b_corr = ustat.correl_func_gaps(
                    effects[effect], behave,
                        years, mint, maxt)

            results += [{'outcome': effect, 'mint': mint, 'maxt': maxt,
                    'sdev': sd,
                    'tscore_corr': t_corr,
                    'behave_corr': b_corr
                        }]
            print(pd.DataFrame(results))

results = pd.DataFrame(results)

### 3) Make graphs
def custom_format(estimates):
    formatted = []
    for i, x in enumerate(estimates):
        if i == 0:
            formatted.append(f"${x:.3f}$")
        elif i == 1:
            formatted.append(f"$[{x:.3f}]$")
        elif i == 2:
            formatted.append(f"$\\{{{x:.3f}\\}}$")
        else:
            formatted.append(f"${x:.3f}$")
    return "\\makecell{" + " \\\\ ".join(formatted) + "}"

results['maxt'] = results.maxt.fillna('4+')
for outset, name in [
        (['Test scores','Study skills','Behaviors'],'shortrun'),
        (['Any CJC', 'Criminal arrest', 'Index crime'],'longrun_cjc'),
        ]:
    for type in ['sdev','tscore_corr','behave_corr']:
        grouped = (
            results.loc[results.outcome.isin(outset)].groupby(
                    ['mint', 'maxt'])[type]
              .apply(custom_format)
              .unstack('maxt')
        )
        latex_str = grouped.to_latex(na_rep="", escape=False)  # escape=False allows commas, decimals
        with open(f"tables/tableA10.tex", "w") as f:
            f.write(latex_str)


