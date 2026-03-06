"""
fake_data_formats.py — Reference only (not run during replication)

This script was used to create data/summary.pkl from the real NCERDC
administrative data. It reads the original analysis .dta file, extracts
summary statistics (means, SDs, min/max, categorical frequencies), and
saves them as a pickle file. The pickle is then used by
simulate_from_summary.py to generate synthetic data with matching
variable names and approximate distributions.

This file is included for transparency; it cannot be run without access
to the original restricted-use data.
"""

import pickle
import pandas as pd

dp = pd.read_stata('/scratch/public/ncrime/analysis_11_01_2021.dta')


summary = {
    'shape': dp.shape,
    'dtypes': dp.dtypes.to_dict(),
    'numeric_stats': dp.describe().loc[['mean', 'std', 'min', 'max']].to_dict(),
    'categoricals': {
        col: dp[col].value_counts(normalize=True).to_dict()
        for col in dp.select_dtypes('object').columns
    },
}


with open('summary.pkl', 'wb') as f:
    pickle.dump(summary, f)

