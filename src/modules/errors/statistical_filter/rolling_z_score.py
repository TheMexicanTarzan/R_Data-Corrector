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
    Detect and correct outliers using Adaptive Rolling Statistics.

    Uses a lagged rolling window to calculate statistics, ensuring that the current
    observation at time t is not included in the window statistics (preventing
    look-forward bias).

    The rolling Z-score is calculated as:
        Z_t = (x_t - ¼_{t-1}) / Ã_{t-1}

    where:
        ¼_{t-1} = rolling mean over [t-window, t-1]
        Ã_{t-1} = rolling std over [t-window, t-1]

    Threshold: Observations are flagged if |Z_t| > k, where k defaults to 3.5
    if not derived from confidence level.

    Window size: Default is 63 trading days (approximately one quarter).

    Args:
        df: Input DataFrame or LazyFrame containing data for a single ticker.
        metadata: LazyFrame containing metadata (unused in this filter but required
                  for signature consistency).
        ticker: Ticker symbol for logging purposes.
        columns: List of column names to apply rolling Z-score filter to.
        date_col: Name of the date column (default: 'm_date').
        confidence: Confidence level (unused, threshold defaults to 3.5).

    Returns:
        tuple containing:
            - Corrected DataFrame/LazyFrame (same type as input)
            - List of dictionaries documenting each correction made
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())

    # Initialize logs
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # Check if date column exists
    if date_col not in schema_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "missing_date_column",
            "message": f"Date column '{date_col}' not found in dataframe"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Filter available columns
    available_cols = [col for col in columns if col in schema_cols]

    if not available_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "no_valid_columns",
            "message": "No valid columns found for rolling Z-score analysis"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Collect data for numpy processing
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    # Get dates for logging
    dates = working_df[date_col].to_list()

    # Default window size (63 trading days ~ 1 quarter)
    window_size = 63

    # Threshold for outlier detection
    threshold = 3.5

    for col in available_cols:
        col_corrections_logged = 0

        # Extract values as numpy array
        values = working_df[col].to_numpy().astype(numpy.float64)

        # Track original null positions
        null_mask = ~numpy.isfinite(values)

        # Calculate lagged rolling statistics
        # We use a shifted window so statistics at time t are based on [t-window, t-1]
        n = len(values)

        if n < window_size + 1:
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "insufficient_data",
                "message": f"Only {n} observations, need at least {window_size + 1} for rolling Z-score"
            })
            continue

        # Calculate rolling mean and std for lagged window
        # For each position i, we want stats from [i-window, i-1]
        rolling_mean = numpy.full(n, numpy.nan)
        rolling_std = numpy.full(n, numpy.nan)

        for i in range(window_size, n):
            # Window is [i-window, i-1], so indices [i-window, i)
            window_values = values[i - window_size:i]

            # Get valid values in window
            valid_window = window_values[numpy.isfinite(window_values)]

            if len(valid_window) >= 10:  # Require at least 10 valid observations
                rolling_mean[i] = numpy.mean(valid_window)
                rolling_std[i] = numpy.std(valid_window, ddof=1)

        # Calculate Z-scores
        z_scores = numpy.full(n, numpy.nan)

        for i in range(window_size, n):
            if numpy.isfinite(values[i]) and numpy.isfinite(rolling_mean[i]) and rolling_std[i] > 0:
                z_scores[i] = (values[i] - rolling_mean[i]) / rolling_std[i]

        # Identify outliers
        outlier_mask = numpy.abs(z_scores) > threshold
        outlier_indices = numpy.where(outlier_mask)[0]

        if len(outlier_indices) == 0:
            continue

        # Create a set for fast lookup
        outlier_set = set(outlier_indices)

        # Get valid indices (non-null and non-outlier)
        valid_mask_for_spline = numpy.isfinite(values) & ~outlier_mask
        valid_indices = numpy.where(valid_mask_for_spline)[0]

        # Impute outliers using cubic spline
        for idx in outlier_indices:
            original_value = float(values[idx])
            z_score_val = float(z_scores[idx])
            mean_val = float(rolling_mean[idx])
            std_val = float(rolling_std[idx])

            # Find valid neighbors for spline interpolation (excluding other outliers)
            prev_valid_indices = []
            prev_valid_values = []
            next_valid_indices = []
            next_valid_values = []

            # Look backward for valid points
            for i in range(idx - 1, -1, -1):
                if i in valid_indices:
                    prev_valid_indices.append(i)
                    prev_valid_values.append(values[i])
                if len(prev_valid_indices) >= 3:
                    break

            # Look forward for valid points
            for i in range(idx + 1, n):
                if i in valid_indices:
                    next_valid_indices.append(i)
                    next_valid_values.append(values[i])
                if len(next_valid_indices) >= 3:
                    break

            # Combine for spline
            spline_indices = prev_valid_indices[::-1] + next_valid_indices
            spline_values = prev_valid_values[::-1] + next_valid_values

            if len(spline_indices) < 3:
                # Fallback to last valid value
                if prev_valid_values:
                    corrected_value = prev_valid_values[0]
                    method = 'last_valid_value'
                elif next_valid_values:
                    corrected_value = next_valid_values[0]
                    method = 'next_valid_value'
                else:
                    # Skip if no valid neighbors
                    if col_corrections_logged < MAX_CORRECTIONS_LOG:
                        logs.append({
                            "ticker": ticker,
                            "date": dates[idx],
                            "column": col,
                            "error_type": "rolling_z_outlier_skipped",
                            "original_value": original_value,
                            "z_score": z_score_val,
                            "rolling_mean": mean_val,
                            "rolling_std": std_val,
                            "method": "skipped_no_valid_neighbors"
                        })
                        col_corrections_logged += 1
                    continue
            else:
                # Apply cubic spline interpolation
                spline = CubicSpline(spline_indices, spline_values)
                corrected_value = float(spline(idx))
                method = 'cubic_spline'

                # Validate corrected value
                if not numpy.isfinite(corrected_value):
                    corrected_value = prev_valid_values[0] if prev_valid_values else next_valid_values[0]
                    method = 'last_valid_value_fallback'

            # Apply correction
            values[idx] = corrected_value

            # Log correction
            if col_corrections_logged < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],
                    "column": col,
                    "error_type": "rolling_z_outlier",
                    "original_value": original_value,
                    "corrected_value": corrected_value,
                    "z_score": z_score_val,
                    "rolling_mean": mean_val,
                    "rolling_std": std_val,
                    "threshold": threshold,
                    "window_size": window_size,
                    "method": method
                })
                col_corrections_logged += 1

        # Restore original null positions
        values[null_mask] = numpy.nan

        # Update working dataframe with corrected column
        working_df = working_df.with_columns(
            polars.Series(name=col, values=values)
        )

    # Join corrected columns back to original lazy frame
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
