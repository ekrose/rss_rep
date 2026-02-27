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

hatch1 = ''
hatch2 = '/'
hatch3 = 'X'
hatch4 = '+'

# Graph of correlation in effects
covs = [('white','Race (white vs. non-white)'),('male','Gender'),('disadv','Econ disadv'),('atrisk','High/low arrest risk')]
results = {}
for covariate, covname in covs:
    results[covname] = {}
    tresids = pd.read_stata("temp/teach_mean_resids_cov{}.dta".format(
            spec, droppval, covariate)).drop('teachid', axis=1)
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
        sdev = ustat.correl_func(d0, d1)
        se = ustat.corr_samp_covar(d0, d1)**0.5
        results[covname][out] = (sdev, se)

labels = ['Test scores', 'Study skills', 'Behaviors']
fig, axes = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.2
for adj, lab, hatch in [(-1.5, 'Race (white vs. non-white)', hatch1), (-0.5, 'Gender', hatch2), (0.5,'Econ disadv', hatch3), (1.5,'High/low arrest risk', hatch4)]:
    axes.bar(x+adj*width, [results[lab][l][0] for l in labels],
            width,
            yerr=[results[lab][l][1]*1.96 for l in labels],
            label=lab,
            hatch=hatch)

axes.set_ylabel('Correlation in short-run effects across groups') 
axes.set_xlabel('Short-run outcome') 
axes.set_xticks(x)
axes.set_ylim(0,1)
axes.set_xticklabels(labels)
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=2)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figure3a.pdf')


# Long runs original
labels = ['Any CJC', 'Criminal arrest','Index crime','Incarceration']
fig, axes = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.2
for adj, lab, hatch in [(-1.5, 'Race (white vs. non-white)', hatch1), (-0.5, 'Gender', hatch2), (0.5,'Econ disadv', hatch3), (1.5,'High/low arrest risk', hatch4)]:
    axes.bar(x+adj*width, [results[lab][l][0] for l in labels],
            width,
            yerr=[results[lab][l][1]*1.96 for l in labels],
            label=lab,
            hatch=hatch)

axes.set_ylabel('Correlation in long-run effects across groups') 
axes.set_xlabel('Long-run outcome') 
axes.set_xticks(x)
axes.set_ylim(0,1)
axes.set_xticklabels(labels)
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=2)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figure3b.pdf')

# Long runs original
labels = ['Graduation','12th grade GPA','College attendance']
fig, axes = plt.subplots(figsize=(8,5))
x = np.arange(len(labels))
width = 0.2
for adj, lab, hatch in [(-1.5, 'Race (white vs. non-white)', hatch1), (-0.5, 'Gender', hatch2), (0.5,'Econ disadv', hatch3), (1.5,'High/low arrest risk', hatch4)]:
    axes.bar(x+adj*width, [results[lab][l][0] for l in labels],
            width,
            yerr=[results[lab][l][1]*1.96 for l in labels],
            label=lab,
            hatch=hatch)

axes.set_ylabel('Correlation in long-run effects across groups') 
axes.set_xlabel('Long-run outcome') 
axes.set_xticks(x)
axes.set_ylim(0,1)
axes.set_xticklabels(labels)
axes.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, ncol=2)
axes.set_facecolor('white')
axes.grid(axis='y', color='grey')
fig.tight_layout()
fig.savefig('figures/figure3c.pdf')


# Any criminal arrest: Graph of 1 SD effects
covs = [('white','Non-white','White'),('male','Female','Male'),('disadv','Not disadv','Disadv'),('atrisk','Low risk','High risk')]
results = {}
for covariate, cov0name, cov1name in covs:
    results[cov0name] = {}
    results[cov1name] = {}
    tresids = pd.read_stata("temp/teach_mean_resids_cov{}.dta".format(
            spec, droppval, covariate)).drop('teachid', axis=1)
    cog0 = tresids.filter(regex='testscores_r0', axis=1).values[:,:]
    cog1 = tresids.filter(regex='testscores_r1', axis=1).values[:,:]
    crime0 = tresids.filter(regex='aoc_crim_r0', axis=1).values[:,:]
    crime1 = tresids.filter(regex='aoc_crim_r1', axis=1).values[:,:]
    behave0 = tresids.filter(regex='behavpca_r0', axis=1).values[:,:]
    behave1 = tresids.filter(regex='behavpca_r1', axis=1).values[:,:]
    study0 = tresids.filter(regex='studypca_r0', axis=1).values[:,:]
    study1 = tresids.filter(regex='studypca_r1', axis=1).values[:,:]   
    for out, d0, d1 in [('Test scores',cog0,cog1),('Study skills',study0,study1),('Behaviors',behave0,behave1)]:
        print('Working on group: {} and outcome: {}'.format(covariate, out))
        effect0 = ustat.sd_effect_func(d0, crime0)
        sd0 = ustat.sd_effect_samp_covar(d0, crime0)**0.5
        results[cov0name][out] = (effect0, sd0)

        effect1 = ustat.sd_effect_func(d1, crime1)
        sd1 = ustat.sd_effect_samp_covar(d1, crime1)**0.5
        results[cov1name][out] = (effect1,sd1)

# Make figure
means = {'Non-white':0.28,'White':0.21, 'Female':0.17,'Male':0.30,'Not disadv':0.16,'Disadv':0.29,'Low risk':0.15,'High risk':0.34}
for out in ['Test scores', 'Behaviors']:
    labels = [(0,'Non-white','White'),(1,'Female','Male'),(2,'Not disadv','Disadv'),(3,'Low risk','High risk')]
    fig, axes = plt.subplots(figsize=(8,5))
    width = 0.3
    for x, cov0name, cov1name in labels:
        axes.bar(x-0.5*width, results[cov0name][out][0],
                        width, yerr=results[cov0name][out][1]*1.96, 
                        color = col_crime)
        
        offset_dirc = -1*(results[cov0name][out][0]<0) + 1*(results[cov0name][out][0]>0)
        axes.annotate('{:4.1f}%'.format(results[cov0name][out][0]/means[cov0name] * 100 ),xy=(x-0.5*width / 2, results[cov0name][out][0]),
        xytext=(-20 , (offset_dirc < 0) * -25 + (offset_dirc > 0) * 15), # X points vertical offset
        textcoords="offset points",
        va='bottom')
        
        offset_dirc = -1*(results[cov1name][out][0]<0) + 1*(results[cov1name][out][0]>0)
        axes.bar(x+0.5*width, results[cov1name][out][0],
                        width, yerr=results[cov1name][out][1]*1.96,
                        color = col_crime, alpha = 0.5)
        axes.annotate('{:4.1f}%'.format( results[cov1name][out][0]/means[cov1name] * 100 ),xy=(x+0.5*width / 2, results[cov1name][out][0]),
        xytext=(0, (offset_dirc < 0) * -25 + (offset_dirc > 0) * 15), # X points vertical offset
        textcoords="offset points",
        va='bottom')

        axes.set_ylabel('Effect of 1 sd increase in\nteacher effects on arrests')
        axes.set_xlabel('Demographic group') 
        axes.set_ylim(-.012,0.004)
        x = [[a-0.5*width,a+0.5*width] for a in range(4)]
        axes.set_xticks([inner for outer in x for inner in outer])
        x = [[a,b] for k, a, b in labels]
        axes.set_xticklabels([inner for outer in x for inner in outer], rotation = 45)
        axes.set_facecolor('white')
        axes.grid(axis='y', color='grey')
        fig.tight_layout()
        if out == "Behaviors":
            fig.savefig('figures/figure4a')
        else:
            fig.savefig('figures/figureA5a')

# Any incarceration: Graph of 1 SD effects
covs = [('white','Non-white','White'),('male','Female','Male'),('disadv','Not disadv','Disadv'),('atrisk','Low risk','High risk')]
results = {}
for covariate, cov0name, cov1name in covs:
    results[cov0name] = {}
    results[cov1name] = {}
    tresids = pd.read_stata("temp/teach_mean_resids_cov{}.dta".format(
            spec, droppval, covariate)).drop('teachid', axis=1)
    cog0 = tresids.filter(regex='testscores_r0', axis=1).values[:,:]
    cog1 = tresids.filter(regex='testscores_r1', axis=1).values[:,:]
    incar0 = tresids.filter(regex='aoc_incar_r0', axis=1).values[:,:]
    incar1 = tresids.filter(regex='aoc_incar_r1', axis=1).values[:,:]
    behave0 = tresids.filter(regex='behavpca_r0', axis=1).values[:,:]
    behave1 = tresids.filter(regex='behavpca_r1', axis=1).values[:,:]
    study0 = tresids.filter(regex='studypca_r0', axis=1).values[:,:]
    study1 = tresids.filter(regex='studypca_r1', axis=1).values[:,:]   
    for out, d0, d1 in [('Test scores',cog0,cog1),('Study skills',study0,study1),('Behaviors',behave0,behave1)]:
        print('Working on group: {} and outcome: {}'.format(covariate, out))
        effect0 = ustat.sd_effect_func(d0, incar0)
        sd0 = ustat.sd_effect_samp_covar(d0, crime0)**0.5
        results[cov0name][out] = (effect0, sd0)

        effect1 = ustat.sd_effect_func(d1, incar1)
        sd1 = ustat.sd_effect_samp_covar(d1, crime1)**0.5
        results[cov1name][out] = (effect1,sd1)

# Make figure
means = {'Non-white':0.11,'White':0.073, 'Female':0.048,'Male':0.13,'Not disadv':0.043,'Disadv':0.12,'Low risk':0.036,'High risk':0.15}
for out in ['Test scores', 'Study skills', 'Behaviors']:
    labels = [(0,'Non-white','White'),(1,'Female','Male'),(2,'Not disadv','Disadv'),(3,'Low risk','High risk')]
    fig, axes = plt.subplots(figsize=(8,5))
    width = 0.3
    for x, cov0name, cov1name in labels:
        axes.bar(x-0.5*width, results[cov0name][out][0],
                        width, yerr=results[cov0name][out][1]*1.96, 
                        color = col_aoc_incar)
        
        offset_dirc = -1*(results[cov0name][out][0]<0) + 1*(results[cov0name][out][0]>0)
        axes.annotate('{:4.1f}%'.format( results[cov0name][out][0]/means[cov0name] * 100 ),xy=(x-0.5*width / 2, results[cov0name][out][0]),
        xytext=(-20 , (offset_dirc < 0) * -25 + (offset_dirc > 0) * 15), # X points vertical offset
        textcoords="offset points",
        va='bottom')
        
        offset_dirc = -1*(results[cov1name][out][0]<0) + 1*(results[cov1name][out][0]>0)
        axes.bar(x+0.5*width, results[cov1name][out][0],
                        width, yerr=results[cov1name][out][1]*1.96,
                        color = col_aoc_incar, alpha = 0.5)
        axes.annotate('{:4.1f}%'.format( results[cov1name][out][0]/means[cov1name] * 100 ),xy=(x+0.5*width / 2, results[cov1name][out][0]),
        xytext=(0, (offset_dirc < 0) * -25 + (offset_dirc > 0) * 15), # X points vertical offset
        textcoords="offset points",
        va='bottom')      
        axes.set_ylabel('Effect of 1 sd increase in\nteacher effects on incarceration')
        axes.set_xlabel('Demographic group') 
        axes.set_ylim(-.012,0.004)
        x = [[a-0.5*width,a+0.5*width] for a in range(4)]
        axes.set_xticks([inner for outer in x for inner in outer])
        x = [[a,b] for k, a, b in labels]
        axes.set_xticklabels([inner for outer in x for inner in outer], rotation = 45)
        axes.set_facecolor('white')
        axes.grid(axis='y', color='grey')
        fig.tight_layout()
        if out == "Behaviors":
            fig.savefig('figures/figure4b')
        else:
            fig.savefig('figures/figureA5b')




