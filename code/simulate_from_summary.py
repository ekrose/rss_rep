import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from pathlib import Path
import os

DATA_DIR = Path(os.environ["PROJECT_DATA_DIR"])


# -----------------------------------------------------------------------
# Hand-specified correlations for the fake-data exercise.
#
# We no longer have access to the restricted-use microdata, so there is
# no empirical correlation matrix to draw on. simulate_from_summary()
# previously drew every column independently from its marginal
# distribution (mean/std/min/max, or category frequencies), which means
# none of the $covdesign covariates carry any real relationship to the
# outcomes (testscores, behavpca, aoc_crim, ...) in the simulated data.
# That's fine for most tables, but it makes Figure A4 (sensitivity of
# the 1-SD teacher-effect estimate to which covariates are controlled
# for) degenerate: adding/removing covariates from the spec does
# essentially nothing, since none of them explain any of the outcome's
# variance.
#
# CORR_OVERRIDES hand-specifies a plausible correlation for a modest set
# of covariate/outcome pairs (magnitudes chosen to be directionally
# sensible, not fit to data). Everything not listed here is left
# uncorrelated (0). This is fed into a Gaussian copula (see
# _build_synthetic_corr / the copula block in simulate_from_summary)
# so the simulated columns keep their original marginal distributions
# but gain this correlation structure, giving Figure A4 non-degenerate
# variation across specifications.
#
# Raw columns feeding the outcomes used in Figure A4 (see preamble.do):
#   testscores (short-run) <- mathscal, readscal
#   behavpca   (short-run) <- lead1_daysabs, lead1_any_discp, lead_grade_rep
#   aoc_crim   (long-run, "Criminal arrest")
# -----------------------------------------------------------------------
CORR_OVERRIDES = {
    # Own persistence: current test scores vs. lagged test scores
    ('mathscal', 'lag1_mathscal'): 0.65,
    ('readscal', 'lag1_readscal'): 0.65,
    ('mathscal', 'lag2_mathscal'): 0.55,
    ('readscal', 'lag2_readscal'): 0.55,
    ('lag1_mathscal', 'lag2_mathscal'): 0.60,
    ('lag1_readscal', 'lag2_readscal'): 0.60,
    ('mathscal', 'readscal'): 0.60,

    # Parental education -> test scores
    ('pared_baormore', 'mathscal'): 0.20,
    ('pared_baormore', 'readscal'): 0.20,
    ('pared_somecol', 'mathscal'): 0.10,
    ('pared_somecol', 'readscal'): 0.10,
    ('pared_hsorless', 'mathscal'): -0.08,
    ('pared_hsorless', 'readscal'): -0.08,
    ('pared_nohs', 'mathscal'): -0.15,
    ('pared_nohs', 'readscal'): -0.15,

    # Disadvantage / limited English -> test scores, crime
    ('disadv', 'mathscal'): -0.25,
    ('disadv', 'readscal'): -0.25,
    ('disadv', 'aoc_crim'): 0.15,
    ('lim_eng', 'readscal'): -0.20,
    ('lim_eng', 'mathscal'): -0.05,

    # Race -> test scores, crime (directional only, not calibrated)
    ('black', 'mathscal'): -0.15,
    ('black', 'readscal'): -0.15,
    ('black', 'aoc_crim'): 0.10,
    ('white', 'mathscal'): 0.10,
    ('white', 'readscal'): 0.10,
    ('white', 'aoc_crim'): -0.08,

    # Gender -> behavior, crime
    ('female', 'aoc_crim'): -0.15,
    ('female', 'lead1_any_discp'): -0.10,

    # AIG (academically/intellectually gifted) -> test scores
    ('aigmath', 'mathscal'): 0.35,
    ('aigread', 'readscal'): 0.35,
    ('aigmath', 'aoc_crim'): -0.05,
    ('aigread', 'aoc_crim'): -0.05,

    # Exceptionality categories -> behavior, test scores, crime
    ('exc_aig', 'mathscal'): 0.20,
    ('exc_aig', 'readscal'): 0.20,
    ('exc_behav', 'lead1_any_discp'): 0.30,
    ('exc_behav', 'lead1_daysabs'): 0.15,
    ('exc_behav', 'aoc_crim'): 0.15,
    ('exc_educ', 'lead1_daysabs'): 0.10,
    ('exc_educ', 'mathscal'): -0.15,
    ('exc_educ', 'readscal'): -0.15,
    ('exc_not', 'aoc_crim'): 0.05,

    # Grade retention -> test scores, own persistence
    ('grade_rep', 'mathscal'): -0.20,
    ('grade_rep', 'readscal'): -0.20,
    ('grade_rep', 'lead_grade_rep'): 0.40,

    # Discipline / attendance history -> behavior outcomes, crime
    ('lag1_any_discp', 'lead1_any_discp'): 0.35,
    ('lag1_any_discp', 'aoc_crim'): 0.20,
    ('lag1_daysabs', 'lead1_daysabs'): 0.35,
    ('lag1_daysabs', 'aoc_crim'): 0.10,

    # Behavioral outcome components -> crime, each other
    ('lead1_any_discp', 'aoc_crim'): 0.25,
    ('lead1_daysabs', 'aoc_crim'): 0.12,
    ('lead_grade_rep', 'aoc_crim'): 0.08,
    ('lead1_any_discp', 'lead1_daysabs'): 0.20,
    ('lead1_any_discp', 'lead_grade_rep'): 0.15,
    ('lead1_daysabs', 'lead_grade_rep'): 0.10,

    # Test scores -> crime
    ('mathscal', 'aoc_crim'): -0.12,
    ('readscal', 'aoc_crim'): -0.10,

    # Study-skills components (homework/freeread/watchtv feed studypca in
    # preamble.do) and additional long-run outcomes. These entries exist
    # mainly to pull the columns into the Gaussian copula so that the
    # teacher-effect factors below (TEACHER_EFFECT_LOADINGS) reach them.
    ('homework', 'freeread'): 0.15,
    ('homework', 'watchtv'): -0.10,
    ('homework', 'mathscal'): 0.15,
    ('freeread', 'readscal'): 0.20,
    ('aoc_any', 'aoc_crim'): 0.60,
    ('aoc_index', 'aoc_crim'): 0.50,
    ('aoc_incar', 'aoc_crim'): 0.45,
    ('grad', 'mathscal'): 0.20,
    ('grad', 'aoc_crim'): -0.15,
    ('gpa_weighted', 'mathscal'): 0.35,
    ('gpa_weighted', 'readscal'): 0.30,
}


# -----------------------------------------------------------------------
# Teacher-effect factors.
#
# The real data contain persistent teacher effects: a teacher's students
# do systematically better or worse on short- and long-run outcomes, and
# these effects are correlated across outcomes. Without them, the entry
# instruments in Tables 5/A6/A7 have a truly zero first stage on the
# simulated data and ivreg2/ivreghdfe cannot even compute the first-stage
# F statistic (the moment covariance matrix is rank deficient, so the
# Kleibergen-Paap statistic is reported as missing).
#
# We therefore add a two-factor teacher random effect to the latent
# Gaussian draws of the copula, BEFORE the marginal transforms:
#   factor 1 ("cognitive")  — loads on test scores and academic outcomes
#   factor 2 ("behavioral") — loads on discipline/attendance and CJC
# The two factors are correlated (TEACHER_FACTOR_CORR). Each column's
# latent draw becomes
#   z' = sqrt(1-s) * z + sqrt(s) * (normalized loading combination),
# with s = TEACHER_EFFECT_SHARE, so every marginal distribution is
# unchanged; only within-teacher dependence is added. Because binary
# columns threshold the copula uniform and continuous columns invert it,
# the teacher component flows through every outcome type.
# -----------------------------------------------------------------------
TEACHER_EFFECT_SHARE = 0.20   # share of latent variance from the teacher
TEACHER_FACTOR_CORR = -0.25   # corr(cognitive, behavioral-problems) factors

# col -> (loading on cognitive factor, loading on behavioral-problems factor)
TEACHER_EFFECT_LOADINGS = {
    'mathscal':        (1.0,  0.0),
    'readscal':        (1.0,  0.0),
    'homework':        (0.5, -0.3),
    'freeread':        (0.5, -0.2),
    'watchtv':         (-0.3, 0.3),
    'lead1_daysabs':   (0.0,  1.0),
    'lead1_any_discp': (0.0,  1.0),
    'lead_grade_rep':  (-0.2, 0.8),
    'aoc_any':         (-0.15, 0.7),
    'aoc_crim':        (-0.15, 0.8),
    'aoc_index':       (-0.15, 0.7),
    'aoc_incar':       (-0.15, 0.7),
    'grad':            (0.6, -0.4),
    'gpa_weighted':    (0.8, -0.2),
}


# -----------------------------------------------------------------------
# Some string categoricals encode "no response" as an explicit blank
# string ("") rather than NaN (e.g. bound_for_code, the survey item
# college_bound is derived from in preamble.do). value_counts() keeps ""
# as a normal category, so simulate_from_summary() reproduces its real
# ~70% blank rate. At n_observations=10,000 that leaves too few rows
# with a real college_bound status once preamble.do's sample
# restrictions and the OVB regressions' fixed effects are applied
# (reghdfe ends up with as many parameters as observations, R2=1.0000,
# and binscatter has no residual variation left to plot -> Figure A3c
# renders as a single point). This caps the blank share for named
# columns and redistributes the rest proportionally across the other
# categories, purely so the small simulated sample has enough
# observations to produce a non-degenerate demonstration figure.
# -----------------------------------------------------------------------
CATEGORICAL_MAX_BLANK_RATE = {
    "bound_for_code": 0.15,
}


def _cap_blank_probability(col, categories, probs):
    """
    If `col` has an override in CATEGORICAL_MAX_BLANK_RATE, cap the
    probability mass on the blank ("") category at that rate and
    redistribute the excess proportionally across the other categories.
    No-op if the column has no override or no blank category.
    """
    max_blank = CATEGORICAL_MAX_BLANK_RATE.get(col)
    if max_blank is None:
        return probs

    blank_idx = np.where(categories == "")[0]
    if len(blank_idx) == 0:
        return probs
    blank_idx = blank_idx[0]

    if probs[blank_idx] <= max_blank:
        return probs

    excess = probs[blank_idx] - max_blank
    other_mask = np.arange(len(probs)) != blank_idx
    other_total = probs[other_mask].sum()
    probs = probs.copy()
    probs[blank_idx] = max_blank
    if other_total > 0:
        probs[other_mask] += probs[other_mask] / other_total * excess
    else:
        probs[other_mask] = excess / other_mask.sum()
    return probs


def _build_synthetic_corr(available_cols):
    """
    Build a full correlation matrix over `available_cols` from
    CORR_OVERRIDES: 1.0 on the diagonal, the specified value for pairs
    named in CORR_OVERRIDES (order-insensitive), 0.0 elsewhere. Columns
    named in CORR_OVERRIDES that aren't present in `available_cols` are
    silently skipped (e.g. if a variable name doesn't match the summary).
    Returns None if fewer than 2 of the named columns are available.
    """
    named_cols = {c for pair in CORR_OVERRIDES for c in pair}
    cols = [c for c in available_cols if c in named_cols]
    if len(cols) < 2:
        return None

    corr = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for (a, b), val in CORR_OVERRIDES.items():
        if a in corr.index and b in corr.columns:
            corr.loc[a, b] = val
            corr.loc[b, a] = val
    return corr


def _nearest_psd_corr(corr, epsilon=1e-8):
    """
    Project a correlation matrix onto the nearest valid (positive
    semi-definite, unit-diagonal) correlation matrix.

    The empirical correlation matrix in summary['corr'] can be slightly
    non-PSD (pairwise-deletion of missing values, floating point noise),
    which np.random.multivariate_normal rejects. Clipping negative
    eigenvalues and rescaling to a unit diagonal fixes this while leaving
    well-behaved matrices essentially unchanged.
    """
    sym = (corr + corr.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(sym)
    eigvals = np.clip(eigvals, epsilon, None)
    reconstructed = (eigvecs * eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(reconstructed))
    d[d == 0] = 1.0
    return reconstructed / np.outer(d, d)


def simulate_from_summary(summary, n_observations=None, random_state=926823):
    """
    Simulate a DataFrame given a summary dict with keys:
      - 'shape': (n_rows, n_cols)
      - 'dtypes': {col: numpy.dtype}
      - 'numeric_stats': {col: {'mean','std','min','max', ...}}
      - 'categoricals': {col: {category: count, ...}}

    Each column's marginal distribution is drawn independently to match
    its summary stats, EXCEPT for the columns named in CORR_OVERRIDES
    (mostly $covdesign covariates and the short/long-run outcomes used
    in Figure A4), which are drawn jointly via a Gaussian copula so they
    carry the hand-specified correlation structure in CORR_OVERRIDES.
    See the comment above CORR_OVERRIDES for why this is necessary.

    Parameters:
    -----------
    summary : dict
        Dictionary containing the summary statistics
    n_observations : int, optional
        Number of observations to simulate. If None, uses the original shape[0] from summary.
    random_state : int, optional
        Random seed for reproducibility
    """
    rng = np.random.default_rng(random_state)

    # Use provided n_observations or default to original shape
    if n_observations is None:
        n_rows, _ = summary["shape"]
    else:
        n_rows = n_observations
    
    dtypes = summary["dtypes"]
    numeric_stats = summary["numeric_stats"]
    categoricals = summary["categoricals"]

    data = {}

    cols = list(dtypes.keys())

    # ------------------------------------------------------------------
    # Gaussian copula: draw jointly-correlated uniforms for the numeric
    # columns named in CORR_OVERRIDES (see above), using a synthetic
    # correlation matrix since we no longer have the real microdata.
    # Each column's own marginal distribution (mean/std/min/max, handled
    # below) is unaffected; only the dependence across columns changes.
    # ------------------------------------------------------------------
    copula_uniforms = {}
    synth_corr = _build_synthetic_corr([c for c in cols if c in numeric_stats])
    if synth_corr is not None:
        corr_mat = _nearest_psd_corr(synth_corr.to_numpy(dtype=float))
        draws = rng.multivariate_normal(
            mean=np.zeros(len(synth_corr)), cov=corr_mat, size=n_rows, method="eigh"
        )

        # Teacher-effect factors (see TEACHER_EFFECT_LOADINGS above).
        # Teacher IDs are assigned below as consecutive blocks of 4 rows
        # (row i belongs to teacher i//4), so the same block structure can
        # be used here, before the marginal transforms.
        n_teachers_eff = int(np.ceil(n_rows / 4))
        factor_cov = np.array([[1.0, TEACHER_FACTOR_CORR],
                               [TEACHER_FACTOR_CORR, 1.0]])
        factors = rng.multivariate_normal(np.zeros(2), factor_cov, size=n_teachers_eff)
        factors_row = np.repeat(factors, 4, axis=0)[:n_rows]
        for j, c in enumerate(synth_corr.columns):
            loadings = TEACHER_EFFECT_LOADINGS.get(c)
            if loadings is None:
                continue
            lc, lb = loadings
            denom = np.sqrt(lc ** 2 + lb ** 2 + 2 * lc * lb * TEACHER_FACTOR_CORR)
            if denom == 0:
                continue
            tau = (lc * factors_row[:, 0] + lb * factors_row[:, 1]) / denom
            draws[:, j] = (np.sqrt(1 - TEACHER_EFFECT_SHARE) * draws[:, j]
                           + np.sqrt(TEACHER_EFFECT_SHARE) * tau)

        uniforms = norm.cdf(draws)
        for j, c in enumerate(synth_corr.columns):
            copula_uniforms[c] = uniforms[:, j]

    for col in cols:
        col_dtype = dtypes[col]
        col_dtype_str = str(col_dtype)

        # 1) CATEGORICALS (object-like with frequency table)
        if col in categoricals:
            counts_dict = categoricals[col]

            # Drop any categories that represent missing values (NaN/NaT/None)
            clean_items = []
            for k, v in counts_dict.items():
                try:
                    if pd.isna(k):
                        continue
                except Exception:
                    if k is None:
                        continue
                clean_items.append((k, v))

            if len(clean_items) == 0:
                # If all categories were missing, use a single synthetic category
                categories = np.array(["_no_missing_category_"], dtype=object)
                counts = np.array([1.0], dtype=float)
            else:
                categories = np.array([k for k, _ in clean_items], dtype=object)
                counts = np.array([v for _, v in clean_items], dtype=float)
            
            # Handle empty categories or invalid counts (after cleaning)
            if len(categories) == 0 or counts.sum() == 0:
                # Use a default category if no valid categories
                data[col] = np.array(["_no_missing_category_"] * n_rows, dtype=object)
            else:
                probs = counts / counts.sum()
                # Ensure no NaN probabilities (probabilities are not stored in the data)
                probs = np.nan_to_num(probs, nan=0.0)
                if probs.sum() == 0:
                    probs = np.ones(len(categories)) / len(categories)  # uniform if all probs invalid
                else:
                    probs = probs / probs.sum()  # renormalize

                probs = _cap_blank_probability(col, categories, probs)

                # sample with same categorical support, approximate frequencies
                sampled = rng.choice(categories, size=n_rows, p=probs)
                data[col] = sampled
            continue

        # 2) NUMERIC (has summary stats)
        if col in numeric_stats:
            stats = numeric_stats[col]
            mean = stats.get("mean", 0.0)
            
            # Replace NaN mean with 0.0
            if pd.isna(mean):
                mean = 0.0
            
            # Check if this is a datetime column (mean is a Timestamp)
            is_datetime = False
            try:
                if isinstance(mean, pd.Timestamp) or hasattr(mean, 'year'):
                    is_datetime = True
            except:
                pass
            
            if is_datetime:
                # Handle datetime columns
                vmin = stats.get("min")
                vmax = stats.get("max")
                try:
                    # Ensure mean is valid
                    if vmin is None or (isinstance(vmin, float) and np.isnan(vmin)):
                        vmin = mean
                    if vmax is None or (isinstance(vmax, float) and np.isnan(vmax)):
                        vmax = mean
                    
                    vmin = pd.to_datetime(vmin) if vmin is not None else mean
                    vmax = pd.to_datetime(vmax) if vmax is not None else mean
                    mean_dt = pd.to_datetime(mean)
                    
                    # Convert to numeric for simulation, then back to datetime
                    min_numeric = vmin.value
                    max_numeric = vmax.value
                    mean_numeric = mean_dt.value
                    std_numeric = max(abs(max_numeric - min_numeric) / 6, 1)  # ensure positive, avoid division by zero
                    
                    # Generate random timestamps
                    arr_numeric = rng.normal(loc=mean_numeric, scale=std_numeric, size=n_rows)
                    arr_numeric = np.clip(arr_numeric, min_numeric, max_numeric)
                    arr = pd.to_datetime(arr_numeric)
                    # Ensure no NaT values
                    arr = arr.fillna(mean_dt)
                    data[col] = arr
                    continue
                except Exception:
                    # Fallback: use mean date or default
                    try:
                        mean_dt = pd.to_datetime(mean)
                    except:
                        mean_dt = pd.to_datetime("2000-01-01")
                    data[col] = pd.Series([mean_dt] * n_rows)
                    continue
            
            # Handle regular numeric columns
            std = stats.get("std", 0.0)
            
            # Replace NaN std with 0.0
            if pd.isna(std):
                std = 0.0
            
            # Safely extract min/max, ensuring they're numeric
            vmin = stats.get("min")
            vmax = stats.get("max")
            
            # Replace NaN min/max
            if vmin is None or (isinstance(vmin, float) and np.isnan(vmin)):
                vmin = None
            if vmax is None or (isinstance(vmax, float) and np.isnan(vmax)):
                vmax = None
            
            # Check if min/max are numeric, otherwise compute from mean/std
            try:
                vmin = float(vmin) if vmin is not None and not np.isnan(vmin) else (mean if std == 0 else mean - 3 * abs(std))
                vmax = float(vmax) if vmax is not None and not np.isnan(vmax) else (mean if std == 0 else mean + 3 * abs(std))
            except (TypeError, ValueError):
                # If conversion fails, use mean-based defaults
                try:
                    mean_float = float(mean) if not np.isnan(mean) else 0.0
                    std_float = float(std) if std is not None and not np.isnan(std) else 0.0
                    vmin = mean_float if std_float == 0 else mean_float - 3 * abs(std_float)
                    vmax = mean_float if std_float == 0 else mean_float + 3 * abs(std_float)
                except (TypeError, ValueError):
                    vmin = 0.0
                    vmax = 1.0
            
            # Ensure mean and std are numeric and not NaN
            try:
                mean = float(mean) if not np.isnan(mean) else 0.0
                std = float(std) if std is not None and not np.isnan(std) else 0.0
            except (TypeError, ValueError):
                mean = 0.0
                std = 0.0
            
            # Ensure vmin and vmax are valid
            if np.isnan(vmin):
                vmin = mean - 3 * abs(std) if std != 0 else mean - 1
            if np.isnan(vmax):
                vmax = mean + 3 * abs(std) if std != 0 else mean + 1

            # Binary indicators (min=0, max=1): simulate as Bernoulli(mean)
            if vmin == 0.0 and vmax == 1.0 and 0 < mean < 1:
                # Ensure at least one success so SD > 0
                p = max(mean, 2.0 / n_rows)
                u = copula_uniforms.get(col)
                if u is not None:
                    # Threshold the UPPER tail (u > 1-p), not the lower
                    # tail: this makes the indicator an increasing
                    # function of the underlying copula normal, so a
                    # positive entry in CORR_OVERRIDES between this
                    # column and another produces a positive empirical
                    # correlation (using u < p would flip the sign).
                    # Marginal probability is unaffected: P(U > 1-p) = p.
                    arr = (u > (1.0 - p)).astype(np.int64)
                else:
                    arr = rng.binomial(1, p, size=n_rows)
                # Safety: if all same, flip one value
                if arr.sum() == 0:
                    arr[0] = 1
                elif arr.sum() == n_rows:
                    arr[0] = 0
                if "float" in col_dtype_str:
                    data[col] = arr.astype(np.float64)
                else:
                    data[col] = arr
                continue

            # Simulate numeric values, ensuring non-zero empirical variability
            if "int" in col_dtype_str and "datetime" not in col_dtype_str:
                # Integer-like: use discrete sampling to guarantee >1 support points
                try:
                    low = int(np.floor(vmin))
                    high = int(np.ceil(vmax))
                except Exception:
                    low, high = -1, 2

                if low == high:
                    # widen artificial support if original range is a single point
                    low -= 1
                    high = low + 2

                u = copula_uniforms.get(col)
                if u is not None:
                    idx = np.floor(u * (high - low + 1)).astype(np.int64)
                    idx = np.clip(idx, 0, high - low)
                    arr = low + idx
                else:
                    arr = rng.integers(low, high + 1, size=n_rows, dtype=np.int64)
                data[col] = arr
            else:
                # Float-like or other numeric: always use positive scale
                if std is None or std == 0 or np.isnan(std):
                    # create artificial spread around mean if original std is zero
                    span = vmax - vmin
                    if not np.isfinite(span) or span == 0:
                        span = max(abs(mean), 1.0)
                        vmin = mean - span / 2.0
                        vmax = mean + span / 2.0
                    scale = max(abs(span) / 6.0, 1e-6)
                else:
                    scale = max(abs(std), 1e-6)

                u = copula_uniforms.get(col)
                if u is not None:
                    arr = norm.ppf(u, loc=mean, scale=scale)
                else:
                    arr = rng.normal(loc=mean, scale=scale, size=n_rows)
                # clip to a finite range
                arr = np.clip(arr, vmin, vmax)

                if "float" in col_dtype_str:
                    data[col] = arr.astype(np.float64)
                else:
                    data[col] = arr
            continue

        # 3) OTHER TYPES / NO SUMMARY AVAILABLE
        # If we reach here, we don't have numeric_stats or categoricals entry.
        # Create a column with simple, non-degenerate support (no missing values).
        if "datetime64" in col_dtype_str:
            # arbitrary fixed timestamp
            default_date = pd.to_datetime("2000-01-01")
            data[col] = pd.Series([default_date] * n_rows)
        elif "bool" in col_dtype_str:
            # random booleans -> non-zero std as long as n_rows > 1
            data[col] = rng.choice([False, True], size=n_rows)
        elif "int" in col_dtype_str:
            # simple discrete distribution over at least two values
            data[col] = rng.integers(0, 2, size=n_rows, dtype=np.int64)
        elif "float" in col_dtype_str:
            # standard normal-like spread
            data[col] = rng.normal(loc=0.0, scale=1.0, size=n_rows).astype(np.float64)
        else:
            # generic object/string fallback
            data[col] = np.array([""] * n_rows, dtype=object)

    # Build DataFrame
    df = pd.DataFrame(data, columns=cols)

    # Enforce dtypes as closely as possible
    for col in cols:
        target_dtype = dtypes[col]
        try:
            # Skip forcing object to avoid breaking categorical strings
            if str(target_dtype) != "object":
                df[col] = df[col].astype(target_dtype)
        except Exception:
            # If casting fails, keep the simulated values as-is
            pass

    # Apply domain constraints for specific numeric variables
    for col in df.columns:
        col_lower = str(col).lower()
        if pd.api.types.is_numeric_dtype(df[col]):
            # Minimum grade is 4
            if col_lower == "grade":
                df.loc[df[col] < 4, col] = 4
                # Ensure non-zero SD: if all collapsed to 4, bump one value
                if df[col].std() == 0 and len(df[col]) > 1:
                    df.iloc[0, df.columns.get_loc(col)] = df.iloc[0, df.columns.get_loc(col)] + 1

            # Minimum year is 1997
            if "year" in col_lower and not col_lower.startswith("nyears"):
                df.loc[df[col] < 1997, col] = 1997
                # Ensure non-zero SD: if all collapsed to 1997, bump one value
                if df[col].std() == 0 and len(df[col]) > 1:
                    df.iloc[0, df.columns.get_loc(col)] = df.iloc[0, df.columns.get_loc(col)] + 1

    # Teacher/school structure:
    # - each teacher appears exactly 4 times
    # - half the teachers appear in 2 schools (2 obs in each school)
    teacher_cols = [
        c for c in df.columns
        if any(tok in str(c).lower() for tok in ["teacher", "tch", "teachid"])
    ]
    school_cols = [
        c for c in df.columns
        if any(tok in str(c).lower() for tok in ["school", "schl"])
    ]

    for t_col in teacher_cols:
        n_rows = len(df)
        if n_rows < 4:
            raise ValueError("Need at least 4 rows to assign teachers with 4 observations each.")
        if n_rows % 4 != 0:
            raise ValueError(
                f"simulate_from_summary requires number of rows to be a multiple of 4 "
                f"to assign exactly 4 observations per teacher; got {n_rows}."
            )

        # Determine how many teachers we need
        n_teachers = n_rows // 4

        # Use sequential integer IDs so teachid stays numeric (required by
        # Stata factor-variable commands like reghdfe ... abs(x#teachid)).
        teacher_ids = np.arange(1, n_teachers + 1)

        # Assign exactly 4 observations per teacher
        new_teacher_values = np.repeat(teacher_ids, 4)
        df[t_col] = new_teacher_values

        # ------------------------------------------------------------------
        # School / grade / year panel structure.
        #
        # Each teacher contributes two teacher-years (rows 0-1 and rows 2-3
        # of their block), and each teacher-year is placed in a
        # school-grade-year CELL. Cells are packed with ~CELL_CAPACITY
        # teacher-years each, so that:
        #   - school-grade-year cells contain multiple teachers (needed for
        #     the leave-out entry instruments of Tables 5/A6/A7 — with the
        #     previous independent draws, cells averaged ~1.4 rows and the
        #     first-stage F statistic of the entry IVs was not computable);
        #   - grade is constant within a teacher-year (as in the real data);
        #   - each teacher's two teacher-years are in different years and
        #     different school-grades, so the leave-out means in preamble.do
        #     (loo_ebar_schgrade / loo_ebar_school) are always defined;
        #   - half of the teachers teach in two schools (2 obs in each),
        #     the rest in one school (unchanged from before).
        # ------------------------------------------------------------------
        if school_cols:
            s_col = school_cols[0]

            GRADES = [4, 5, 6, 7, 8]
            YEARS = [2000, 2001, 2002, 2003, 2004, 2005]
            CELL_CAPACITY = 3

            n_teacher_years = 2 * n_teachers
            n_schools = max(
                int(np.ceil(n_teacher_years / (len(GRADES) * len(YEARS) * CELL_CAPACITY))),
                20,
            )

            # Per-school cycler over (grade, year) slots, each repeated
            # CELL_CAPACITY times, in random order.
            school_slots = {}
            for s in range(n_schools):
                slots = [(g, y) for g in GRADES for y in YEARS] * CELL_CAPACITY
                rng.shuffle(slots)
                school_slots[s] = slots

            def _take_slot(school, exclude_year=None):
                """Pop the next (grade, year) slot, preferring a different year."""
                slots = school_slots[school]
                for k, (g, y) in enumerate(slots):
                    if exclude_year is None or y != exclude_year:
                        return slots.pop(k)
                # School exhausted of other years: borrow from the full grid
                g = int(rng.choice(GRADES))
                y = int(rng.choice([yy for yy in YEARS if yy != exclude_year] or YEARS))
                return (g, y)

            # One row block per teacher (4 rows per teacher)
            indices = df.index.to_numpy()
            teacher_blocks = np.split(indices, n_teachers)

            # Choose half the teachers (floor) to be multi-school
            n_multi = n_teachers // 2
            multi_teacher_idx = set(rng.choice(np.arange(n_teachers), size=n_multi, replace=False))

            grade_cols = [c for c in df.columns if str(c).lower() == "grade"]
            year_cols = [c for c in df.columns if str(c).lower() == "year"]
            g_col = grade_cols[0] if grade_cols else None
            y_col = year_cols[0] if year_cols else None

            # Record each teacher-year's cell for the tenure variables below:
            # cell_of_ty[(teacher_index, 0 or 1)] = (school, grade, year)
            cell_of_ty = {}

            for i, block_idx in enumerate(teacher_blocks):
                s1 = i % n_schools
                if i in multi_teacher_idx and n_schools >= 2:
                    s2 = (s1 + 1 + (i // n_schools)) % n_schools
                    if s2 == s1:
                        s2 = (s1 + 1) % n_schools
                else:
                    s2 = s1

                g1, y1 = _take_slot(s1)
                g2, y2 = _take_slot(s2, exclude_year=y1)

                # schlcode is a string column in the real data
                df.loc[block_idx[:2], s_col] = str(s1 + 1)
                df.loc[block_idx[2:], s_col] = str(s2 + 1)
                if g_col is not None:
                    df.loc[block_idx[:2], g_col] = g1
                    df.loc[block_idx[2:], g_col] = g2
                if y_col is not None:
                    df.loc[block_idx[:2], y_col] = y1
                    df.loc[block_idx[2:], y_col] = y2

                cell_of_ty[(i, 0)] = (s1, g1, y1)
                cell_of_ty[(i, 1)] = (s2, g2, y2)

            # Synchronize school_fe (numeric FE used by Stata) with
            # the school structure just assigned via s_col.
            fe_col_name = "school_fe"
            if fe_col_name in df.columns and fe_col_name != s_col:
                unique_schools_assigned = df[s_col].unique()
                school_to_fe = {s: i + 1 for i, s in enumerate(unique_schools_assigned)}
                df[fe_col_name] = df[s_col].map(school_to_fe)

            # --------------------------------------------------------------
            # Coherent tenure variables for the entry IVs (Tables 5/A6/A7).
            # These are ordinary columns in the real data; here they are
            # rebuilt to be consistent with the cell structure above:
            #   - nyears_schgrd: age of the school-grade cell (constant
            #     within cell, >= 3 for ~85% of cells so the entry-IV
            #     sample condition "nyears_schgrd >= 3" usually holds);
            #   - nyears_teach_schgrd: teacher's tenure in the school-grade
            #     (0 = new entrant, ~45% of teacher-years);
            #   - nyears_teach_school: teacher's tenure in the school
            #     (0 only when also new to the school-grade).
            # --------------------------------------------------------------
            cell_age = {}
            for (i, half), (s, g, y) in cell_of_ty.items():
                key = (s, g)
                if key not in cell_age:
                    cell_age[key] = int(rng.integers(3, 15)) if rng.random() < 0.85 else int(rng.integers(1, 3))

            has_t_schgrd = "nyears_teach_schgrd" in df.columns
            has_t_school = "nyears_teach_school" in df.columns
            has_schgrd = "nyears_schgrd" in df.columns

            if has_t_schgrd or has_t_school or has_schgrd:
                for i, block_idx in enumerate(teacher_blocks):
                    for half, rows in ((0, block_idx[:2]), (1, block_idx[2:])):
                        s, g, y = cell_of_ty[(i, half)]
                        age = cell_age[(s, g)]
                        entrant = rng.random() < 0.45
                        tenure_sg = 0 if entrant else int(rng.integers(1, max(age, 2)))
                        tenure_s = tenure_sg + int(rng.integers(0, 4))
                        if entrant and rng.random() < 0.7:
                            tenure_s = 0
                        if has_schgrd:
                            df.loc[rows, "nyears_schgrd"] = age
                        if has_t_schgrd:
                            df.loc[rows, "nyears_teach_schgrd"] = tenure_sg
                        if has_t_school:
                            df.loc[rows, "nyears_teach_school"] = tenure_s

        # Enforce subject structure within each teacher so that
        # preamble.do's "drop if nteachhr_years < 2" drops zero rows and
        # the Chetty VAM leave-one-out procedure always has a valid
        # complement year. (Year structure — two distinct years per
        # teacher, rows 0-1 vs rows 2-3 — is guaranteed by the cell
        # assignment above.)
        subject_cols = [
            c for c in df.columns
            if any(tok in str(c).lower() for tok in ["subject", "subj"])
        ]

        for subj_col in subject_cols:
            all_subjects = df[subj_col].dropna().unique()
            if len(all_subjects) == 0:
                continue
            for i, block_idx in enumerate(teacher_blocks):
                # All 4 rows get the same subject
                subj = rng.choice(all_subjects)
                df.loc[block_idx, subj_col] = subj


    # Ensure student-level ID columns (e.g. mastid) are unique integers.
    # In the real data each row is a distinct student-teacher-year-subject
    # observation, so the student identifier must be unique per row.
    student_id_cols = [
        c for c in df.columns
        if any(tok in str(c).lower() for tok in ["mastid"])
    ]
    for sid_col in student_id_cols:
        df[sid_col] = np.arange(1, n_rows + 1).astype(df[sid_col].dtype)

    # twin_id identifies twin pairs: each twin_id should be shared by
    # exactly 2 observations.  Assign pair IDs so that rows 0-1 share
    # twin_id 1, rows 2-3 share twin_id 2, etc.
    twin_cols = [
        c for c in df.columns if str(c).lower() == "twin_id"
    ]
    for tw_col in twin_cols:
        pair_ids = np.repeat(np.arange(1, n_rows // 2 + 1), 2)
        if n_rows % 2 == 1:
            pair_ids = np.append(pair_ids, pair_ids[-1] + 1)
        df[tw_col] = pair_ids.astype(df[tw_col].dtype)

    # Check that each *relevant* numeric variable has non-zero standard deviation.
    # Treat pure identifier columns (e.g. teachid_*) as categorical for this purpose
    # and exclude them from the SD requirement.
    numeric_stds = df.std(numeric_only=True)
    id_like_cols = [
        c for c in numeric_stds.index
        if any(tok in str(c).lower() for tok in ["teachid", "teacher", "cmb_teachid", "mastid", "twin_id"])
    ]
    zero_std = numeric_stds[
        (numeric_stds == 0)
        & ~numeric_stds.index.isin(id_like_cols)
    ]
    if not zero_std.empty:
        raise ValueError(
            "The following numeric variables have zero standard deviation in the "
            f"simulated data: {', '.join(zero_std.index.astype(str))}"
        )

    # As a safety check, assert that we did not create any missing values
    if df.isna().values.any():
        raise ValueError("simulate_from_summary produced missing values, which should never happen.")

    return df


def export_to_stata(df: pd.DataFrame, path: str, version: int = 118) -> None:
    """
    Export a DataFrame to Stata as robustly as possible.

    - Cleans column names to be valid Stata variable names
    - Avoids index export
    - Uses modern .dta format with support for long strings
    """
    df_out = df.copy()

    # 1) Validate (but DO NOT change) column names for Stata compatibility
    #    We only warn if there might be issues; names stay exactly as in `summary`.
    invalid = []
    for col in df_out.columns:
        name = str(col)
        if len(name) > 32 or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            invalid.append(name)
    if invalid:
        print("Warning: the following variable names may not be valid in Stata and")
        print("could cause errors on write/read or be modified by Stata itself:")
        for name in invalid:
            print(f"  {name}")

    # 2) Make dtypes Stata-friendly (without touching column names)
    for col in df_out.columns:
        s = df_out[col]
        if s.dtype == "bool":
            # Stata has no boolean type; use 0/1
            df_out[col] = s.astype("uint8")
        # Ensure all object columns are strings (no mixed types)
        elif s.dtype == "object":
            df_out[col] = s.astype("string").astype("object")

    # 3) Write to Stata
    df_out.to_stata(
        path,
        write_index=False,
        version=version,       # modern Stata format
        # let pandas decide which string columns should become strL
        convert_strl=None,
        time_stamp=None,       # avoid embedding current timestamp
    )


if __name__ == "__main__":
    # Adjust this path if needed. Run from the project root so this
    # resolves to <repo>/temp/summary.pkl (temp/ is gitignored).
    summary_path = Path("temp/summary.pkl")

    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
    
    # Example: simulate with custom number of observations
    df_sim_custom = simulate_from_summary(summary, n_observations=10000)
    print("Simulated shape (custom 20k):", df_sim_custom.shape)
    
    # Verify no missing values
    missing_counts = df_sim_custom.isna().sum()
    total_missing = missing_counts.sum()
    if total_missing == 0:
        print("\n✓ Confirmed: No missing values in simulated data")
    else:
        print(f"\n⚠ Warning: Found {total_missing} missing values across columns:")
        print(missing_counts[missing_counts > 0])
    
    print("\nFirst few rows:")
    print(df_sim_custom.head())

    # If present, coerce all *watchtv-related* variables to numeric 0/1
    # indicators before export so that Stata sees them as numeric rather than
    # string. This covers columns like watchtv, lead1_watchtv, etc.
    watchtv_like_cols = [
        c for c in df_sim_custom.columns
        if "watchtv" in str(c).lower()
    ]
    # Backwards-compatible alias in case older code referred to `watchtv_likes`
    watchtv_likes = watchtv_like_cols

    if watchtv_like_cols:
        yes_values = {"yes", "y", "1", "true", "t"}
        no_values = {"no", "n", "0", "false", "f"}

        def _map_watchtv(v):
            if v in yes_values:
                return 1
            if v in no_values:
                return 0
            return np.nan

        for col_name in watchtv_like_cols:
            col = df_sim_custom[col_name]
            if col.dtype == "object":
                s = col.astype("string").str.lower().str.strip()
                mapped = s.map(_map_watchtv)
                # If mapping produced any NaN, fall back to 0 for those
                df_sim_custom[col_name] = mapped.fillna(0).astype("uint8")
            else:
                # If already numeric/bool, coerce to an unsigned byte
                df_sim_custom[col_name] = df_sim_custom[col_name].astype("uint8")

    # Export to Stata
    export_path = DATA_DIR / "analysis.dta"
    export_to_stata(df_sim_custom, export_path)
    print(f"\nWrote Stata file to {export_path}")
