import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline


def mad_filter(
    df: Union[polars.DataFrame, polars.LazyFrame],
    metadata: polars.LazyFrame,
    ticker: str,
    columns: list[str],
    date_col: str = "m_date",
    confidence: float = 0.01
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Modified Z-score (Robust Statistics).

    Uses Median Absolute Deviation (MAD) to identify outliers in a robust manner
    that is resistant to extreme values. The Modified Z-score is calculated as:

        Mod_Z = 0.6745 * (x - median) / MAD

    where MAD = median(|x - median|). The constant 0.6745 makes the MAD
    consistent with the standard deviation for normally distributed data.

    Threshold: Observations are flagged as outliers if |Mod_Z| > 3.5.

    Args:
        df: Input DataFrame or LazyFrame containing data for a single ticker.
        metadata: LazyFrame containing metadata (unused in this filter but required
                  for signature consistency).
        ticker: Ticker symbol for logging purposes.
        columns: List of column names to apply MAD filter to.
        date_col: Name of the date column (default: 'm_date').
        confidence: Confidence level for outlier detection (unused, threshold is 3.5).

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
            "message": "No valid columns found for MAD analysis"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Collect data for numpy processing
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    # Get dates for logging
    dates = working_df[date_col].to_list()

    # Threshold for outlier detection
    threshold = 3.5

    # Consistency constant for MAD (makes MAD consistent with std for normal data)
    consistency_constant = 0.6745

    for col in available_cols:
        col_corrections_logged = 0

        # Extract values as numpy array
        values = working_df[col].to_numpy().astype(numpy.float64)

        # Track original null positions
        null_mask = ~numpy.isfinite(values)

        # Get valid (non-null) values
        valid_mask = numpy.isfinite(values)
        valid_indices = numpy.where(valid_mask)[0]

        if len(valid_indices) < 5:
            # Not enough data for MAD calculation
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "insufficient_data",
                "message": f"Only {len(valid_indices)} valid observations, need at least 5 for MAD"
            })
            continue

        valid_values = values[valid_mask]

        # Calculate Median
        median_val = numpy.median(valid_values)

        # Calculate MAD (Median Absolute Deviation)
        abs_deviations = numpy.abs(valid_values - median_val)
        mad_val = numpy.median(abs_deviations)

        # Avoid division by zero
        if mad_val == 0:
            # All values are identical or near-identical - no outliers possible
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "zero_mad",
                "message": "MAD is zero, all values are identical or near-identical"
            })
            continue

        # Calculate Modified Z-scores for all valid values
        modified_z_scores = consistency_constant * (valid_values - median_val) / mad_val

        # Identify outliers
        outlier_mask_valid = numpy.abs(modified_z_scores) > threshold
        outlier_indices_valid = numpy.where(outlier_mask_valid)[0]

        if len(outlier_indices_valid) == 0:
            continue

        # Map outlier indices back to original array
        outlier_original_indices = valid_indices[outlier_indices_valid]

        # Create a set for fast lookup
        outlier_set = set(outlier_original_indices)

        # Impute outliers using cubic spline
        for idx in outlier_original_indices:
            original_value = float(values[idx])
            modified_z = float(modified_z_scores[numpy.where(valid_indices == idx)[0][0]])

            # Find valid neighbors for spline interpolation (excluding other outliers)
            prev_valid_indices = []
            prev_valid_values = []
            next_valid_indices = []
            next_valid_values = []

            # Look backward for valid points
            for i in range(idx - 1, -1, -1):
                if i in valid_indices and i not in outlier_set:
                    prev_valid_indices.append(i)
                    prev_valid_values.append(values[i])
                if len(prev_valid_indices) >= 3:
                    break

            # Look forward for valid points
            for i in range(idx + 1, len(values)):
                if i in valid_indices and i not in outlier_set:
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
                            "error_type": "mad_outlier_skipped",
                            "original_value": original_value,
                            "modified_z_score": modified_z,
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
                    "error_type": "mad_outlier",
                    "original_value": original_value,
                    "corrected_value": corrected_value,
                    "modified_z_score": modified_z,
                    "median": float(median_val),
                    "mad": float(mad_val),
                    "threshold": threshold,
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
