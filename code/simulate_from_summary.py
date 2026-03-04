import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pathlib import Path  
import os  

DATA_DIR = Path(os.environ["PROJECT_DATA_DIR"])


def simulate_from_summary(summary, n_observations=None, random_state=926823):
    """
    Simulate a DataFrame given a summary dict with keys:
      - 'shape': (n_rows, n_cols)
      - 'dtypes': {col: numpy.dtype}
      - 'numeric_stats': {col: {'mean','std','min','max', ...}}
      - 'categoricals': {col: {category: count, ...}}
    
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
            if "grade" in col_lower:
                df.loc[df[col] < 4, col] = 4
                # Ensure non-zero SD: if all collapsed to 4, bump one value
                if df[col].std() == 0 and len(df[col]) > 1:
                    df.iloc[0, df.columns.get_loc(col)] = df.iloc[0, df.columns.get_loc(col)] + 1

            # Minimum year is 1997
            if "year" in col_lower:
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

        # Base teacher IDs from the existing data in this column
        base_ids = df[t_col].dropna().unique()
        if len(base_ids) == 0:
            base_ids = np.array([f"T{i}" for i in range(1, n_teachers + 1)], dtype=object)

        teacher_ids = []
        while len(teacher_ids) < n_teachers:
            teacher_ids.extend(list(base_ids))
        teacher_ids = np.array(teacher_ids[:n_teachers], dtype=object)

        # Assign exactly 4 observations per teacher
        new_teacher_values = np.repeat(teacher_ids, 4)
        df[t_col] = new_teacher_values

        # If we also have a school column, enforce that half of the teachers
        # are observed in two schools (2 obs in each), and the rest in a single school.
        if school_cols:
            s_col = school_cols[0]
            schools = df[s_col].dropna().unique()
            if len(schools) == 0:
                schools = np.array(["S1", "S2"], dtype=object)
            elif len(schools) == 1:
                schools = np.array([schools[0], "S2"], dtype=object)

            # One row block per teacher (4 rows per teacher)
            indices = df.index.to_numpy()
            teacher_blocks = np.split(indices, n_teachers)

            # Choose half the teachers (floor) to be multi-school
            n_multi = n_teachers // 2
            multi_teacher_idx = set(rng.choice(np.arange(n_teachers), size=n_multi, replace=False))

            for i, (tid, block_idx) in enumerate(zip(teacher_ids, teacher_blocks)):
                if i in multi_teacher_idx and len(schools) >= 2:
                    # Two schools, two obs in each
                    s1, s2 = rng.choice(schools, size=2, replace=False)
                    df.loc[block_idx[:2], s_col] = s1
                    df.loc[block_idx[2:], s_col] = s2
                else:
                    # Single school (all 4 obs same school)
                    s = rng.choice(schools)
                    df.loc[block_idx, s_col] = s

        # If we also have a subject column, ensure each teacher has at least
        # two observations in the same subject. Teachers can be observed in
        # only some subjects, but for any teacher there must exist at least
        # one subject with count >= 2.
        subject_cols = [
            c for c in df.columns
            if any(tok in str(c).lower() for tok in ["subject", "subj"])
        ]
        for subj_col in subject_cols:
            # For each teacher, find the maximum count over subjects
            pair_counts = (
                df.groupby([t_col, subj_col]).size().rename("n")
            )
            if pair_counts.empty:
                continue

            max_per_teacher = pair_counts.groupby(level=0).max()
            bad_teachers = max_per_teacher[max_per_teacher < 2].index.to_list()

            if not bad_teachers:
                continue

            all_subjects = df[subj_col].dropna().unique()
            if len(all_subjects) == 0:
                continue

            for teacher_id in bad_teachers:
                idxs = df.index[df[t_col] == teacher_id]
                if len(idxs) < 2:
                    # Should not happen because we already enforced >=2 obs
                    continue

                # Choose a target subject for this teacher: either the one
                # they already have (if any) or from the global pool.
                current_subjects = df.loc[idxs, subj_col].dropna().unique()
                if len(current_subjects) > 0:
                    target_subject = rng.choice(current_subjects)
                else:
                    target_subject = rng.choice(all_subjects)

                # Assign the first two observations for this teacher to the
                # target subject so that subject count for this teacher is >= 2.
                two_idxs = idxs.to_list()[:2]
                df.loc[two_idxs, subj_col] = target_subject


    # Check that each *relevant* numeric variable has non-zero standard deviation.
    # Treat pure identifier columns (e.g. teachid_*) as categorical for this purpose
    # and exclude them from the SD requirement.
    numeric_stds = df.std(numeric_only=True)
    id_like_cols = [
        c for c in numeric_stds.index
        if any(tok in str(c).lower() for tok in ["teachid", "teacher", "cmb_teachid"])
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
    # Adjust this path if needed
    summary_path = Path("/Users/shemtov/Downloads/summary.pkl")

    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
    
    # Example: simulate with custom number of observations
    df_sim_custom = simulate_from_summary(summary, n_observations=10000)
    print("Simulated shape (custom 10k):", df_sim_custom.shape)
    
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
