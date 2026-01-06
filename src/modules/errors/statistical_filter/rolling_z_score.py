import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline


def rolling_z_score(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.01
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Adaptive Rolling Statistics (Optimized).
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # 1. Validation checks (Identical to original)
    if date_col not in schema_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "missing_date_column",
            "message": f"Date column '{date_col}' not found"
        })
        return (working_lf if is_lazy else working_lf.collect(), logs)

    available_cols = [col for col in columns if col in schema_cols]
    if not available_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "no_valid_columns",
            "message": "No valid columns found"
        })
        return (working_lf if is_lazy else working_lf.collect(), logs)

    # 2. Materialize only needed columns, sorted by date
    # We collect here because interpolation (Scipy) requires materialized data
    # and Polars window functions work best when we know the data is physically sorted.
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    # Configurations
    window_size = 63
    threshold = 3.5  # Fixed threshold as per original logic
    min_periods = 10  # Matches original "len(valid_window) >= 10"

    # We will build expressions to update the dataframe
    # But since we need to log specific values, we iterate columns to process logic
    # This is still fast because the heavy lifting (stats) is vectorized.

    dates = working_df[date_col].to_list()

    for col in available_cols:
        # 3. Vectorized Rolling Stats Calculation
        # shift(1) ensures we don't include current time t in the window (Look-forward bias prevention)
        # rolling_mean/std are optimized Polars ops.
        stats = working_df.select([
            polars.col(col),
            polars.col(col).shift(1).rolling_mean(window_size, min_periods=min_periods).alias("mean"),
            polars.col(col).shift(1).rolling_std(window_size, min_periods=min_periods).alias("std")
        ])

        # Calculate Z-scores vectors
        # Using numpy for the final outlier mask calculation to easily interface with Scipy later
        col_values = stats[col].to_numpy().copy()
        rolling_mean = stats["mean"].to_numpy()  # These are just read, so copies aren't strictly needed
        rolling_std = stats["std"].to_numpy()

        # Avoid division by zero or NaN issues
        with numpy.errstate(invalid='ignore', divide='ignore'):
            z_scores = (col_values - rolling_mean) / rolling_std

        # Create masks
        valid_stats_mask = numpy.isfinite(rolling_mean) & (rolling_std > 0)
        # Outliers: |Z| > threshold AND valid stats existed
        outlier_mask = (numpy.abs(z_scores) > threshold) & valid_stats_mask & numpy.isfinite(col_values)

        outlier_indices = numpy.where(outlier_mask)[0]

        if len(outlier_indices) == 0:
            continue

        # 4. Interpolation & Logging
        # We perform interpolation only if outliers exist.

        # Mask outliers in the data to treat them as missing for the spline
        # Original logic: "valid_mask_for_spline = numpy.isfinite(values) & ~outlier_mask"
        clean_values_for_fit = col_values.copy()
        clean_values_for_fit[outlier_mask] = numpy.nan

        # Get indices of valid data for training the spline
        valid_indices = numpy.where(numpy.isfinite(clean_values_for_fit))[0]
        valid_values = clean_values_for_fit[valid_indices]

        # Use Scipy CubicSpline on ALL valid data
        # This is faster and smoother than finding 3 neighbors iteratively
        if len(valid_indices) > 3:
            try:
                cs = CubicSpline(valid_indices, valid_values)
                # Predict only at outlier locations
                corrected_values = cs(outlier_indices)
                method_used = "cubic_spline"
            except Exception:
                # Fallback if spline fails (e.g. strict geometric issues)
                method_used = "fallback_nearest"
                corrected_values = [valid_values[numpy.abs(valid_indices - idx).argmin()] for idx in outlier_indices]
        else:
            # Not enough data to interpolate
            continue

        col_corrections_logged = 0

        # Apply corrections and Log
        # We iterate only the outliers (usually very few rows)
        for i, idx in enumerate(outlier_indices):
            original_val = float(col_values[idx])
            new_val = float(corrected_values[i])

            # Validation: ensure we didn't generate a NaN or infinity
            if not numpy.isfinite(new_val):
                # Fallback to nearest valid
                nearest_idx = valid_indices[numpy.abs(valid_indices - idx).argmin()]
                new_val = float(clean_values_for_fit[nearest_idx])
                method_used = "last_valid_value_fallback"

            # Log
            if col_corrections_logged < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],
                    "column": col,
                    "error_type": "rolling_z_outlier",
                    "original_value": original_val,
                    "corrected_value": new_val,
                    "z_score": float(z_scores[idx]),
                    "rolling_mean": float(rolling_mean[idx]),
                    "rolling_std": float(rolling_std[idx]),
                    "threshold": threshold,
                    "window_size": window_size,
                    "method": method_used
                })
                col_corrections_logged += 1

            # Update the main array
            col_values[idx] = new_val

        # Update the column in the working DataFrame
        working_df = working_df.with_columns(
            polars.Series(name=col, values=col_values)
        )

    # 5. Join back to original structure
    # Since we sorted and collected working_df, we need to ensure alignment if passed back as Lazy

    # Select only the corrected columns
    corrected_cols_df = working_df.select(available_cols)

    # Reconstruct the result LazyFrame
    # We join based on a row index to ensure we map corrections back to the exact source rows
    result_lf = (
        df.lazy()
        .sort(date_col)
        .with_row_index("_join_idx")
        .drop(available_cols)  # Drop old versions of columns
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