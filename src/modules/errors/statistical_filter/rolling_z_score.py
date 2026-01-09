import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.stats import norm  # Required for dynamic threshold calculation


def rolling_z_score(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.0001,
        shared_data: dict = None  # Unused - for interface consistency with cross-sectional filters
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Adaptive Rolling Statistics with Dynamic Thresholds.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # 1. Validation checks
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
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    # Configurations
    window_size = 63
    min_periods = 10

    # -- Dynamic Threshold Calculation --
    # Calculate the Z-score cutoff based on the confidence level (Two-tailed)
    # 99% Confidence (0.01) -> 0.5% tail -> ~2.576 Sigma
    try:
        safe_confidence = max(1e-10, min(confidence, 1.0 - 1e-10))
        threshold = norm.ppf(1 - (safe_confidence / 2))
    except Exception:
        threshold = 3.5  # Fallback heuristic

    dates = working_df[date_col].to_list()

    # Collect ALL outliers across all columns, then sort by severity
    all_outlier_logs = []

    for col in available_cols:
        # 3. Vectorized Rolling Stats Calculation
        # shift(1) ensures we don't include current time t in the window
        # (Look-forward bias prevention)
        stats = working_df.select([
            polars.col(col),
            polars.col(col).shift(1).rolling_mean(window_size, min_periods=min_periods).alias("mean"),
            polars.col(col).shift(1).rolling_std(window_size, min_periods=min_periods).alias("std")
        ])

        # Calculate Z-scores vectors
        # Using .copy() to ensure writable memory
        col_values = stats[col].to_numpy().copy()
        rolling_mean = stats["mean"].to_numpy()
        rolling_std = stats["std"].to_numpy()

        # Avoid division by zero or NaN issues
        with numpy.errstate(invalid='ignore', divide='ignore'):
            z_scores = (col_values - rolling_mean) / rolling_std

        # Create masks
        valid_stats_mask = numpy.isfinite(rolling_mean) & (rolling_std > 0)

        # Outliers: |Z| > Dynamic Threshold AND valid stats existed
        outlier_mask = (numpy.abs(z_scores) > threshold) & valid_stats_mask & numpy.isfinite(col_values)
        outlier_indices = numpy.where(outlier_mask)[0]

        if len(outlier_indices) == 0:
            continue

        # 4. Interpolation & Logging
        # Mask outliers in the data to treat them as missing for the spline
        clean_values_for_fit = col_values.copy()
        clean_values_for_fit[outlier_mask] = numpy.nan

        # Get indices of valid data for training the spline
        valid_indices = numpy.where(numpy.isfinite(clean_values_for_fit))[0]
        valid_values = clean_values_for_fit[valid_indices]

        # Use Scipy CubicSpline on ALL valid data
        if len(valid_indices) > 3:
            try:
                cs = CubicSpline(valid_indices, valid_values)
                corrected_values = cs(outlier_indices)
                method_used = "cubic_spline"
            except Exception:
                method_used = "fallback_nearest"
                corrected_values = [valid_values[numpy.abs(valid_indices - idx).argmin()] for idx in outlier_indices]
        else:
            continue

        # Apply corrections and Log
        for i, idx in enumerate(outlier_indices):
            original_val = float(col_values[idx])
            new_val = float(corrected_values[i])

            # Validation: ensure we didn't generate a NaN or infinity
            if not numpy.isfinite(new_val):
                nearest_idx = valid_indices[numpy.abs(valid_indices - idx).argmin()]
                new_val = float(clean_values_for_fit[nearest_idx])
                method_used = "last_valid_value_fallback"

            # Collect ALL outliers with severity for later sorting
            all_outlier_logs.append({
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
                "confidence_alpha": confidence,
                "window_size": window_size,
                "method": method_used,
                "_severity": abs(float(z_scores[idx]))  # For sorting by most anomalous
            })

            # Update the main array
            col_values[idx] = new_val

        # Update the column in the working DataFrame
        working_df = working_df.with_columns(
            polars.Series(name=col, values=col_values)
        )

    # Sort by severity (highest |z-score| first) and keep top K most anomalous
    all_outlier_logs.sort(key=lambda x: x["_severity"], reverse=True)
    for log_entry in all_outlier_logs[:MAX_CORRECTIONS_LOG]:
        del log_entry["_severity"]  # Remove internal sorting field
        logs.append(log_entry)

    # 5. Join back to original structure
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