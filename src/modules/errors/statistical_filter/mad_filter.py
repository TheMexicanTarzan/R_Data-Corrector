import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.stats import norm  # Needed for critical value calculation


def mad_filter(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.001
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Modified Z-score with Dynamic Thresholds.
    """
    # 1. Setup & Validation
    is_lazy = isinstance(df, polars.LazyFrame)
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
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()
    dates = working_df[date_col].to_list()

    # -- Dynamic Threshold Calculation --
    # 1. We treat 'confidence' as alpha (significance level). E.g. 0.01 = 99% Confidence.
    # 2. We need the two-tailed critical value.
    #    Example: For 0.01, we want the point where 99.5% of data is to the left.
    #    norm.ppf(0.995) ~= 2.576
    try:
        # Safety clamp to prevent math errors if confidence is 0 or negative
        safe_confidence = max(1e-10, min(confidence, 1.0 - 1e-10))
        threshold = norm.ppf(1 - (safe_confidence / 2))
    except Exception:
        # Fallback if calculation fails
        threshold = 3.5

    # Consistency constant for MAD (makes MAD consistent with std for normal data)
    consistency_constant = 0.6745

    # 3. Vectorized Loop per Column
    for col in available_cols:
        # -- Step A: Calculate Stats in Polars (Fast) --
        stats = working_df.select([
            polars.col(col).median().alias("median"),
            (polars.col(col) - polars.col(col).median()).abs().median().alias("mad")
        ])

        median_val = stats["median"][0]
        mad_val = stats["mad"][0]

        if median_val is None or mad_val is None:
            logs.append({"ticker": ticker, "column": col, "error_type": "insufficient_data",
                         "message": "Cannot calc stats (too few values)"})
            continue

        if mad_val == 0:
            logs.append({"ticker": ticker, "column": col, "error_type": "zero_mad",
                         "message": "MAD is zero (identical values)"})
            continue

        # -- Step B: Calculate Z-Scores Vectorized --
        col_values = working_df[col].to_numpy().astype(numpy.float64).copy()

        with numpy.errstate(invalid='ignore', divide='ignore'):
            mod_z_scores = consistency_constant * (col_values - median_val) / mad_val

        # Identify outliers: |Z| > Dynamic Threshold
        outlier_mask = (numpy.abs(mod_z_scores) > threshold) & numpy.isfinite(col_values)
        outlier_indices = numpy.where(outlier_mask)[0]

        if len(outlier_indices) == 0:
            continue

        # -- Step C: Optimized Interpolation --
        clean_values_for_fit = col_values.copy()
        clean_values_for_fit[outlier_mask] = numpy.nan

        valid_indices = numpy.where(numpy.isfinite(clean_values_for_fit))[0]
        valid_values = clean_values_for_fit[valid_indices]

        if len(valid_indices) < 4:
            continue

        try:
            cs = CubicSpline(valid_indices, valid_values)
            corrected_values_array = cs(outlier_indices)
            method_used = "cubic_spline"
        except Exception:
            method_used = "fallback_nearest"
            idx_in_valid = numpy.searchsorted(valid_indices, outlier_indices)
            idx_in_valid = numpy.clip(idx_in_valid, 0, len(valid_indices) - 1)
            corrected_values_array = valid_values[idx_in_valid]

        col_corrections_logged = 0

        # -- Step D: Apply & Log --
        for i, idx in enumerate(outlier_indices):
            original_val = float(col_values[idx])
            new_val = float(corrected_values_array[i])
            z_score_val = float(mod_z_scores[idx])

            if not numpy.isfinite(new_val):
                nearest_idx = valid_indices[numpy.abs(valid_indices - idx).argmin()]
                new_val = float(clean_values_for_fit[nearest_idx])
                method_used = "last_valid_value_fallback"

            col_values[idx] = new_val

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
                    "confidence_alpha": confidence,
                    "method": method_used
                })
                col_corrections_logged += 1

        # -- Step E: Update DataFrame --
        working_df = working_df.with_columns(
            polars.Series(name=col, values=col_values)
        )

    # 4. Join Back to LazyFrame
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