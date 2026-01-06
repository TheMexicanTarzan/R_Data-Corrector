import polars as pl
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline


def mad_filter(
        df: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.01
) -> tuple[Union[pl.DataFrame, pl.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Modified Z-score (Optimized).
    """
    # 1. Setup & Validation
    is_lazy = isinstance(df, pl.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    schema_cols = set(working_lf.collect_schema().names())
    logs = []
    MAX_CORRECTIONS_LOG = 50

    if date_col not in schema_cols:
        logs.append({"ticker": ticker, "error_type": "missing_date", "message": f"Date column '{date_col}' not found"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    available_cols = [col for col in columns if col in schema_cols]
    if not available_cols:
        logs.append({"ticker": ticker, "error_type": "no_valid_cols", "message": "No valid columns found"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 2. Materialize Data
    # Sort once by date to ensure interpolation works correctly
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()
    dates = working_df[date_col].to_list()

    # Constants
    threshold = 3.5
    consistency_constant = 0.6745

    # 3. Vectorized Loop per Column
    for col in available_cols:
        # -- Step A: Calculate Stats in Polars (Fast) --
        # We need median and MAD. Polars handles nulls automatically in aggregations.
        stats = working_df.select([
            pl.col(col).median().alias("median"),
            (pl.col(col) - pl.col(col).median()).abs().median().alias("mad")
        ])

        median_val = stats["median"][0]
        mad_val = stats["mad"][0]

        # Validation checks
        if median_val is None or mad_val is None:
            logs.append({"ticker": ticker, "column": col, "error_type": "insufficient_data",
                         "message": "Cannot calc stats (too few values)"})
            continue

        if mad_val == 0:
            logs.append({"ticker": ticker, "column": col, "error_type": "zero_mad",
                         "message": "MAD is zero (identical values)"})
            continue

        # -- Step B: Calculate Z-Scores Vectorized --
        # Get the column as a writable numpy array (Fix for read-only error)
        col_values = working_df[col].to_numpy().astype(np.float64).copy()

        # Calculate Mod_Z = 0.6745 * (x - median) / MAD
        # Use np.errstate to suppress warnings for NaNs (which we handle later)
        with np.errstate(invalid='ignore', divide='ignore'):
            mod_z_scores = consistency_constant * (col_values - median_val) / mad_val

        # Identify outliers: |Z| > 3.5 AND value is not NaN
        outlier_mask = (np.abs(mod_z_scores) > threshold) & np.isfinite(col_values)
        outlier_indices = np.where(outlier_mask)[0]

        if len(outlier_indices) == 0:
            continue

        # -- Step C: Optimized Interpolation --
        # Prepare data for fitting the spline: Treat outliers as 'missing' to predict them
        # We mask the outliers so the spline learns the 'clean' shape

        # 1. Create a clean Y vector for training
        clean_values_for_fit = col_values.copy()
        clean_values_for_fit[outlier_mask] = np.nan

        # 2. Get valid X (indices) and Y (values)
        valid_indices = np.where(np.isfinite(clean_values_for_fit))[0]
        valid_values = clean_values_for_fit[valid_indices]

        # We need at least 4 points to fit a Cubic Spline comfortably
        if len(valid_indices) < 4:
            continue

        # 3. Fit Spline Once
        try:
            cs = CubicSpline(valid_indices, valid_values)
            # Predict only at outlier locations
            corrected_values_array = cs(outlier_indices)
            method_used = "cubic_spline"
        except Exception:
            # Fallback to nearest neighbor if spline fails (e.g., flat lines)
            method_used = "fallback_nearest"
            # Vectorized nearest neighbor lookup
            # Find index in valid_indices closest to each outlier index
            idx_in_valid = np.searchsorted(valid_indices, outlier_indices)
            idx_in_valid = np.clip(idx_in_valid, 0, len(valid_indices) - 1)
            corrected_values_array = valid_values[idx_in_valid]

        col_corrections_logged = 0

        # -- Step D: Apply & Log --
        for i, idx in enumerate(outlier_indices):
            original_val = float(col_values[idx])
            new_val = float(corrected_values_array[i])
            z_score_val = float(mod_z_scores[idx])

            # Validation: Splines can sometimes shoot to infinity
            if not np.isfinite(new_val):
                # Fallback to simple nearest valid value
                nearest_idx = valid_indices[np.abs(valid_indices - idx).argmin()]
                new_val = float(clean_values_for_fit[nearest_idx])
                method_used = "last_valid_value_fallback"

            # Apply correction
            col_values[idx] = new_val

            # Log
            if col_corrections_logged < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],
                    "column": col,
                    "error_type": "mad_outlier",
                    "original_value": original_val,
                    "corrected_value": new_val,
                    "modified_z_score": z_score_val,
                    "median": float(median_val),
                    "mad": float(mad_val),
                    "threshold": threshold,
                    "method": method_used
                })
                col_corrections_logged += 1

        # -- Step E: Update DataFrame --
        working_df = working_df.with_columns(
            pl.Series(name=col, values=col_values)
        )

    # 4. Join Back to LazyFrame (Standard Pattern)
    corrected_cols_df = working_df.select(available_cols)

    result_lf = (
        df.lazy()
        .sort(date_col)
        .with_row_index("_join_idx")
        .drop(available_cols)
        .join(
            corrected_cols_df.lazy().with_row_index("_join_idx"),
            on="_join_idx",
            how="left"
        )
        .drop("_join_idx")
    )

    if is_lazy:
        return result_lf, logs
    else:
        return result_lf.collect(), logs