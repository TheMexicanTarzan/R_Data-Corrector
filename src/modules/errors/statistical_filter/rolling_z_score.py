import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.stats import norm

MAX_CORRECTIONS_LOG = 5

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
    Optimized for vectorization and parallel execution.
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
        # 99% Confidence (0.01) -> 0.5% tail -> ~2.576 Sigma
        safe_confidence = max(1e-10, min(confidence, 1.0 - 1e-10))
        threshold = norm.ppf(1 - (safe_confidence / 2))
    except Exception:
        threshold = 3.5

    # 3. Vectorized Rolling Stats Calculation
    sorted_lf = working_lf.sort(date_col)

    # Generate expressions for all columns at once
    mean_exprs = [
        polars.col(c).shift(1).rolling_mean(window_size, min_periods=min_periods).alias(f"{c}_mean")
        for c in available_cols
    ]
    std_exprs = [
        polars.col(c).shift(1).rolling_std(window_size, min_periods=min_periods).alias(f"{c}_std")
        for c in available_cols
    ]

    # Collect data, means, and stds in one shot
    computed_df = sorted_lf.select(
        [date_col] + available_cols + mean_exprs + std_exprs
    ).collect()

    dates = computed_df[date_col].to_list()

    # 4. Move to NumPy for Vectorized Z-Score Math
    # Extract matrices: (N_rows x N_cols)
    vals_mat = computed_df.select(available_cols).to_numpy().copy()
    means_mat = computed_df.select([f"{c}_mean" for c in available_cols]).to_numpy()
    stds_mat = computed_df.select([f"{c}_std" for c in available_cols]).to_numpy()

    # Compute Z-scores for the entire matrix at once
    with numpy.errstate(invalid='ignore', divide='ignore'):
        z_scores_mat = (vals_mat - means_mat) / stds_mat

    # Create outlier mask (boolean matrix)
    valid_stats_mask = numpy.isfinite(means_mat) & (stds_mat > 0)
    outlier_mask = (numpy.abs(z_scores_mat) > threshold) & valid_stats_mask & numpy.isfinite(vals_mat)

    # Identify columns that actually have outliers
    cols_with_outliers = numpy.where(outlier_mask.any(axis=0))[0]

    all_outlier_logs = []

    # 5. Iterative Correction (Only on affected columns)
    for col_idx in cols_with_outliers:
        col_name = available_cols[col_idx]
        col_outlier_mask = outlier_mask[:, col_idx]
        outlier_indices = numpy.where(col_outlier_mask)[0]

        # Prepare data for Spline
        col_values = vals_mat[:, col_idx]
        clean_values_for_fit = col_values.copy()
        clean_values_for_fit[col_outlier_mask] = numpy.nan

        valid_indices = numpy.where(numpy.isfinite(clean_values_for_fit))[0]

        # We need at least 4 points for cubic spline
        if len(valid_indices) <= 3:
            continue

        valid_values = clean_values_for_fit[valid_indices]

        # Fit Spline
        try:
            cs = CubicSpline(valid_indices, valid_values)
            corrected_values = cs(outlier_indices)
            method_used = "cubic_spline"
        except Exception:
            method_used = "fallback_nearest"
            corrected_values = []
            for idx in outlier_indices:
                nearest_loc = numpy.abs(valid_indices - idx).argmin()
                corrected_values.append(valid_values[nearest_loc])
            corrected_values = numpy.array(corrected_values)

        # Apply corrections and log
        for k, idx in enumerate(outlier_indices):
            original_val = float(col_values[idx])
            new_val = float(corrected_values[k])

            if not numpy.isfinite(new_val):
                nearest_idx = valid_indices[numpy.abs(valid_indices - idx).argmin()]
                new_val = float(clean_values_for_fit[nearest_idx])
                method_used = "last_valid_value_fallback"

            # Update matrix in-place
            vals_mat[idx, col_idx] = new_val

            z_score_val = float(z_scores_mat[idx, col_idx])
            all_outlier_logs.append({
                "ticker": ticker,
                "date": dates[idx],
                "column": col_name,
                "error_type": "rolling_z_outlier",
                "original_value": original_val,
                "corrected_value": new_val,
                "z_score": z_score_val,
                "rolling_mean": float(means_mat[idx, col_idx]),
                "rolling_std": float(stds_mat[idx, col_idx]),
                "threshold": threshold,
                "confidence_alpha": confidence,
                "window_size": window_size,
                "method": method_used,
                "_severity": abs(z_score_val)
            })

    # 6. Logging
    if all_outlier_logs:
        all_outlier_logs.sort(key=lambda x: x["_severity"], reverse=True)
        for log_entry in all_outlier_logs[:MAX_CORRECTIONS_LOG]:
            del log_entry["_severity"]
            logs.append(log_entry)

    # 7. Reconstruction (FIXED)
    # We remove 'orient="col"' to allow standard Polars inference
    # or construct via dictionary to be explicitly robust.
    # Method: Dictionary construction (Robust against orient deprecations)
    corrected_data_dict = {
        col: vals_mat[:, i] for i, col in enumerate(available_cols)
    }
    corrected_df = polars.DataFrame(corrected_data_dict)

    # We join based on row index
    final_lf = (
        sorted_lf
        .with_row_index("_join_idx")
        .drop(available_cols)  # Drop old columns
        .join(
            corrected_df.lazy().with_row_index("_join_idx"),
            on="_join_idx",
            how="left"
        )
        .drop("_join_idx")
    )

    if is_lazy:
        return final_lf, logs
    else:
        return final_lf.collect(), logs