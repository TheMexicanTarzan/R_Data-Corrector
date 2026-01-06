import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline
from arch import arch_model


def garch_residuals(
    df: Union[polars.DataFrame, polars.LazyFrame],
    metadata: polars.LazyFrame,
    ticker: str,
    columns: list[str],
    date_col: str = "m_date",
    confidence: float = 0.01
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using GARCH(1,1) volatility clustering.

    Fits a GARCH(1,1) model with Student's T-distribution to handle fat tails.
    Calculates standardized residuals and flags observations
    where |z_t| > 3.5. Flagged values are imputed using cubic spline interpolation.

    Constraint: Ensures alpha + beta < 1 for stationarity.

    Args:
        df: Input DataFrame or LazyFrame containing market data for a single ticker.
        metadata: LazyFrame containing metadata (unused in this filter but required
                  for signature consistency).
        ticker: Ticker symbol for logging purposes.
        columns: List of column names to apply GARCH filter to.
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
            "message": "No valid columns found for GARCH analysis"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Collect data for numpy processing (GARCH requires eager evaluation)
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    # Get dates for logging
    dates = working_df[date_col].to_list()

    # Threshold for outlier detection
    threshold = 3.5

    for col in available_cols:
        col_corrections_logged = 0

        # Extract values as numpy array
        values = working_df[col].to_numpy().astype(numpy.float64)

        # Track original null positions
        null_mask = ~numpy.isfinite(values)

        # Get valid (non-null) values and their indices
        valid_mask = numpy.isfinite(values)
        valid_indices = numpy.where(valid_mask)[0]

        if len(valid_indices) < 100:
            # Not enough data for GARCH estimation
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "insufficient_data",
                "message": f"Only {len(valid_indices)} valid observations, need at least 100 for GARCH"
            })
            continue

        # Calculate returns for GARCH model (percentage changes)
        valid_values = values[valid_mask]
        returns = numpy.diff(valid_values) / valid_values[:-1] * 100

        # Handle any inf/nan in returns
        returns = numpy.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        if len(returns) < 100:
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "insufficient_returns",
                "message": f"Only {len(returns)} returns available, need at least 100"
            })
            continue

        try:
            # Fit GARCH(1,1) model with Student's T distribution
            model = arch_model(
                returns,
                vol='Garch',
                p=1,
                q=1,
                dist='t',
                rescale=True
            )

            # Fit the model
            result = model.fit(disp='off', show_warning=False)

            # Extract parameters
            params = result.params
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)

            # Check stationarity constraint
            if alpha + beta >= 1:
                logs.append({
                    "ticker": ticker,
                    "column": col,
                    "error_type": "non_stationary_garch",
                    "message": f"GARCH model non-stationary: alpha + beta = {alpha + beta:.4f} >= 1",
                    "alpha": float(alpha),
                    "beta": float(beta)
                })
                continue

            conditional_volatility = result.conditional_volatility
            std_resid = result.resid / (conditional_volatility + 1e-8)

            # Map standardized residuals back to original indices
            # Returns correspond to indices [1, 2, ..., n-1] of valid_indices
            # So std_resid[i] corresponds to valid_indices[i+1]

            # Find outlier indices in returns space
            outlier_mask_returns = numpy.abs(std_resid) > threshold
            outlier_indices_returns = numpy.where(outlier_mask_returns)[0]

            if len(outlier_indices_returns) == 0:
                continue

            # Map back to original values array indices
            # std_resid[i] corresponds to valid_indices[i+1]
            outlier_original_indices = valid_indices[outlier_indices_returns + 1]

            # Impute outliers using cubic spline
            for idx in outlier_original_indices:
                original_value = float(values[idx])
                std_resid_value = float(std_resid[numpy.where(valid_indices[1:] == idx)[0][0]]) if idx in valid_indices[1:] else None

                # Find valid neighbors for spline interpolation
                prev_valid_indices = []
                prev_valid_values = []
                next_valid_indices = []
                next_valid_values = []

                # Look backward for valid points
                for i in range(idx - 1, -1, -1):
                    if i in valid_indices and i not in outlier_original_indices:
                        prev_valid_indices.append(i)
                        prev_valid_values.append(values[i])
                    if len(prev_valid_indices) >= 3:
                        break

                # Look forward for valid points
                for i in range(idx + 1, len(values)):
                    if i in valid_indices and i not in outlier_original_indices:
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
                    else:
                        # Skip if no valid previous values
                        if col_corrections_logged < MAX_CORRECTIONS_LOG:
                            logs.append({
                                "ticker": ticker,
                                "date": dates[idx],
                                "column": col,
                                "error_type": "garch_outlier_skipped",
                                "original_value": original_value,
                                "standardized_residual": std_resid_value,
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
                        "error_type": "garch_outlier",
                        "original_value": original_value,
                        "corrected_value": corrected_value,
                        "standardized_residual": std_resid_value,
                        "threshold": threshold,
                        "alpha": float(alpha),
                        "beta": float(beta),
                        "method": method
                    })
                    col_corrections_logged += 1

            # Restore original null positions
            values[null_mask] = numpy.nan

            # Update working dataframe with corrected column
            working_df = working_df.with_columns(
                polars.Series(name=col, values=values)
            )

        except Exception as exc:
            logs.append({
                "ticker": ticker,
                "column": col,
                "error_type": "garch_fit_error",
                "message": str(exc)
            })
            continue

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
