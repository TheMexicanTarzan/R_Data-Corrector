import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.stats import norm
import gc

MAX_CORRECTIONS_LOG = 5

# MEMORY OPTIMIZATION: Process columns in chunks to avoid creating huge matrices
MAX_COLS_PER_CHUNK = 10


def rolling_z_score(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.0001,
        shared_data: dict = None
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Adaptive Rolling Statistics with Dynamic Thresholds.

    MEMORY OPTIMIZATION: Process columns in chunks instead of all at once to reduce
    peak memory usage. Also explicitly delete intermediate arrays.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # 1. Validation and Schema Check
    schema_cols = set(working_lf.collect_schema().names())
    logs = []

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

    # 2. Configuration & Dynamic Threshold
    window_size = 21
    min_periods = 10

    try:
        safe_confidence = max(1e-10, min(confidence, 1.0 - 1e-10))
        threshold = norm.ppf(1 - (safe_confidence / 2))
    except Exception:
        threshold = 3.5

    # 3. Sort and get base data (only date column initially)
    sorted_lf = working_lf.sort(date_col)

    # MEMORY OPTIMIZATION: Collect only what we need - date column first
    base_df = sorted_lf.select([date_col]).collect()
    dates = base_df[date_col].to_list()
    n_rows = len(dates)
    del base_df

    # Dictionary to store corrected columns
    corrected_columns = {}
    all_outlier_logs = []

    # 4. MEMORY OPTIMIZATION: Process columns in chunks
    for chunk_start in range(0, len(available_cols), MAX_COLS_PER_CHUNK):
        chunk_end = min(chunk_start + MAX_COLS_PER_CHUNK, len(available_cols))
        chunk_cols = available_cols[chunk_start:chunk_end]

        # Generate expressions for this chunk only
        mean_exprs = [
            polars.col(c).shift(1).rolling_mean(window_size, min_periods=min_periods).alias(f"{c}_mean")
            for c in chunk_cols
        ]
        std_exprs = [
            polars.col(c).shift(1).rolling_std(window_size, min_periods=min_periods).alias(f"{c}_std")
            for c in chunk_cols
        ]

        # Collect only this chunk's data
        chunk_df = sorted_lf.select(chunk_cols + mean_exprs + std_exprs).collect()

        # Extract to numpy arrays
        vals_mat = chunk_df.select(chunk_cols).to_numpy()
        means_mat = chunk_df.select([f"{c}_mean" for c in chunk_cols]).to_numpy()
        stds_mat = chunk_df.select([f"{c}_std" for c in chunk_cols]).to_numpy()

        # Free the DataFrame memory immediately
        del chunk_df

        # Compute Z-scores
        with numpy.errstate(invalid='ignore', divide='ignore'):
            z_scores_mat = (vals_mat - means_mat) / stds_mat

        # Create outlier mask
        valid_stats_mask = numpy.isfinite(means_mat) & (stds_mat > 0)
        outlier_mask = (numpy.abs(z_scores_mat) > threshold) & valid_stats_mask & numpy.isfinite(vals_mat)

        # Process columns with outliers
        cols_with_outliers = numpy.where(outlier_mask.any(axis=0))[0]

        for local_col_idx in cols_with_outliers:
            col_name = chunk_cols[local_col_idx]
            col_outlier_mask = outlier_mask[:, local_col_idx]
            outlier_indices = numpy.where(col_outlier_mask)[0]

            col_values = vals_mat[:, local_col_idx].copy()
            clean_values_for_fit = col_values.copy()
            clean_values_for_fit[col_outlier_mask] = numpy.nan

            valid_indices = numpy.where(numpy.isfinite(clean_values_for_fit))[0]

            if len(valid_indices) <= 3:
                corrected_columns[col_name] = col_values
                continue

            valid_values = clean_values_for_fit[valid_indices]

            # Fit Spline
            try:
                cs = CubicSpline(valid_indices, valid_values)
                corrected_values = cs(outlier_indices)
                method_used = "cubic_spline"
            except Exception:
                method_used = "fallback_nearest"
                corrected_values = numpy.array([
                    valid_values[numpy.abs(valid_indices - idx).argmin()]
                    for idx in outlier_indices
                ])

            # Apply corrections and log (limited)
            log_count = 0
            for k, idx in enumerate(outlier_indices):
                original_val = float(col_values[idx])
                new_val = float(corrected_values[k])

                if not numpy.isfinite(new_val):
                    nearest_idx = valid_indices[numpy.abs(valid_indices - idx).argmin()]
                    new_val = float(clean_values_for_fit[nearest_idx])
                    method_used = "last_valid_value_fallback"

                col_values[idx] = new_val

                # Only log first few for memory efficiency
                if log_count < MAX_CORRECTIONS_LOG * 2:
                    z_score_val = float(z_scores_mat[idx, local_col_idx])
                    all_outlier_logs.append({
                        "ticker": ticker,
                        "date": dates[idx],
                        "column": col_name,
                        "error_type": "rolling_z_outlier",
                        "original_value": original_val,
                        "corrected_value": new_val,
                        "z_score": z_score_val,
                        "threshold": threshold,
                        "method": method_used,
                        "_severity": abs(z_score_val)
                    })
                    log_count += 1

            corrected_columns[col_name] = col_values

        # Store unchanged columns from this chunk
        for local_col_idx, col_name in enumerate(chunk_cols):
            if col_name not in corrected_columns:
                corrected_columns[col_name] = vals_mat[:, local_col_idx].copy()

        # MEMORY FIX: Explicitly delete chunk arrays
        del vals_mat, means_mat, stds_mat, z_scores_mat, outlier_mask, valid_stats_mask

    # 5. Truncate logs by severity
    if all_outlier_logs:
        all_outlier_logs.sort(key=lambda x: x["_severity"], reverse=True)
        for log_entry in all_outlier_logs[:MAX_CORRECTIONS_LOG]:
            del log_entry["_severity"]
            logs.append(log_entry)
        del all_outlier_logs

    # 6. Reconstruction - build corrected DataFrame
    corrected_df = polars.DataFrame(corrected_columns)
    del corrected_columns

    # Join with original sorted LazyFrame
    final_lf = (
        sorted_lf
        .with_row_index("_join_idx")
        .drop(available_cols)
        .join(
            corrected_df.lazy().with_row_index("_join_idx"),
            on="_join_idx",
            how="left"
        )
        .drop("_join_idx")
    )

    del corrected_df
    gc.collect()

    if is_lazy:
        return final_lf, logs
    else:
        return final_lf.collect(), logs