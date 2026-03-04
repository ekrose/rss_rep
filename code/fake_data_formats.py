

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

