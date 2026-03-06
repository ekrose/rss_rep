"""
Simulate data/teacher_covars.dta for teacher_chars.do (Tables A1, A12).

This creates a teacher-year panel of teacher characteristics that merges
onto the analysis data. The real file is built from raw NCERDC
administrative data and is not included in the replication archive;
this script generates a synthetic substitute.

Required variables in teacher_covars.dta:
    teachid      – teacher identifier (merge key)
    year         – year (merge key)
    gender       – string: "F" or "M"
    ethnic       – string: "B" (Black) or "W" (White)
    bdate        – Stata %td date: teacher birth date (used to compute age)
    educ_lvl_cd  – numeric: education level code (>=5 means Masters+)
    score_normed – numeric: average test score of teacher's students (standardized)
    relative_pay – numeric: teacher pay relative to peers (standardized)

Usage:
    python code/simulate_teacher_covars.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import os

rng = np.random.default_rng(8675309)

# ---------------------------------------------------------------------------
# 1) Read the teacher-year pairs from the main analysis data
# ---------------------------------------------------------------------------
data_dir = Path(os.environ["PROJECT_DATA_DIR"])
resids = pd.read_stata(data_dir / "analysis.dta", columns=["teachid", "year"])
resids = resids.drop_duplicates().sort_values(["teachid", "year"]).reset_index(drop=True)

n = len(resids)
n_teachers = resids["teachid"].nunique()
print(f"Building teacher_covars for {n} teacher-year rows ({n_teachers} teachers)")

# ---------------------------------------------------------------------------
# 2) Assign time-invariant teacher characteristics (one draw per teacher)
# ---------------------------------------------------------------------------
teacher_ids = resids["teachid"].unique()

# Gender: ~75% female (typical elementary/middle school)
gender_draw = rng.choice(["F", "M"], size=n_teachers, p=[0.75, 0.25])

# Ethnicity: ~20% Black, 80% White
ethnic_draw = rng.choice(["B", "W"], size=n_teachers, p=[0.20, 0.80])

# Birth year: teachers born between 1950 and 1985
birth_years = rng.integers(1950, 1986, size=n_teachers)

# Birth date as Stata numeric date: days since 1960-01-01
# Pick a random day in the birth year and convert to Stata date
stata_epoch = datetime(1960, 1, 1)
bdates_stata = []
for by in birth_years:
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 29))
    dt = datetime(by, m, d)
    bdates_stata.append((dt - stata_epoch).days)
bdates_stata = np.array(bdates_stata, dtype=np.float64)

# Education level: codes 1-7, with ~40% having Masters+ (>=5)
educ_probs = [0.02, 0.08, 0.15, 0.35, 0.25, 0.10, 0.05]
educ_draw = rng.choice(range(1, 8), size=n_teachers, p=educ_probs)

teacher_chars = pd.DataFrame({
    "teachid": teacher_ids,
    "gender": gender_draw,
    "ethnic": ethnic_draw,
    "bdate": bdates_stata,
    "birth_year": birth_years,
    "educ_lvl_cd": educ_draw,
})

# ---------------------------------------------------------------------------
# 3) Merge onto teacher-year panel and add time-varying characteristics
# ---------------------------------------------------------------------------
df = resids.merge(teacher_chars, on="teachid", how="left")

# score_normed: standardized average test score (N(0,1) with small teacher RE)
teacher_re = rng.normal(0, 0.3, size=n_teachers)
re_map = dict(zip(teacher_ids, teacher_re))
df["score_normed"] = df["teachid"].map(re_map) + rng.normal(0, 0.7, size=n)

# relative_pay: standardized pay (increases with experience/education)
df["relative_pay"] = (
    rng.normal(0, 0.5, size=n)
    + 0.3 * (df["educ_lvl_cd"] - 4)
    + 0.05 * (df["year"] - df["birth_year"] - 25)
)
# Standardize
df["relative_pay"] = (df["relative_pay"] - df["relative_pay"].mean()) / df["relative_pay"].std()

# Drop birth_year helper column (not needed in final output)
df = df.drop(columns=["birth_year"])

# ---------------------------------------------------------------------------
# 4) Save as Stata file
# ---------------------------------------------------------------------------
df["teachid"] = df["teachid"].astype(np.int32)
df["year"] = df["year"].astype(np.int32)
df["educ_lvl_cd"] = df["educ_lvl_cd"].astype(np.int8)
df["score_normed"] = df["score_normed"].astype(np.float64)
df["relative_pay"] = df["relative_pay"].astype(np.float64)

out_path = "data/teacher_covars.dta"
df.to_stata(out_path, write_index=False, version=118)
print(f"Wrote {out_path}: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"  Teachers: {df['teachid'].nunique()}")
print(f"  Years: {sorted(df['year'].unique())}")
print(f"  Gender: {df['gender'].value_counts().to_dict()}")
print(f"  Ethnic: {df['ethnic'].value_counts().to_dict()}")
print(f"  Educ >= 5: {(df['educ_lvl_cd'] >= 5).mean():.1%}")
