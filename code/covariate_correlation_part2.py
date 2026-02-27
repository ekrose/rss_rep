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
### 0) Load
##################################### 

# Load mean covariates 
covs = pd.read_stata("temp/teach_mean_covars.dta")

# Load sigma
sigma = pd.read_stata("temp/sigma.dta").values
assert sigma.shape[0] == sigma.shape[1] == (covs.shape[1]-3)


#####################################
### 1) Compute
##################################### 

# Group by teacher
grouped = {
    j: g.set_index('obs').filter(regex='^_x', axis=1)
    for j, g in covs.groupby('teachid')
}

# Compute correlation for each teacher
def compute_teacher_mean(j):
    X = grouped[j].values
    X_sigma = X @ sigma
    dot_matrix = X_sigma @ X.T
    upper_triangle = dot_matrix[np.triu_indices_from(dot_matrix, k=1)]
    return upper_triangle.mean() if upper_triangle.size > 0 else np.nan

# Add correlations
teach_means = []
for idx, j in tqdm(enumerate(covs.teachid.unique())):
	teach_means += [compute_teacher_mean(j)]


#####################################
### 2) Save
##################################### 

# Report result
print("Resulting average bias for variance {}".format(np.mean(teach_means)))
with open("tables/in_text_citations.txt", "w") as f:
    f.write("Average bias for variance {}".format(np.mean(teach_means)))


