import polars as pl
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.stats import t
from arch import arch_model


def garch_residuals(
        df: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.01
) -> tuple[Union[pl.DataFrame, pl.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using GARCH(1,1) (Performance Optimized).
    """
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

    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()
    dates = working_df[date_col].to_list()

    # Pre-extract all columns as numpy arrays upfront
    col_arrays = {col: working_df[col].to_numpy().astype(np.float64, copy=True)
                  for col in available_cols}

    modified_cols = {}

    for col in available_cols:
        col_values = col_arrays[col]
        valid_mask = np.isfinite(col_values)
        valid_indices = np.flatnonzero(valid_mask)

        if len(valid_indices) < 100:
            logs.append({"ticker": ticker, "column": col, "error_type": "insufficient_data",
                         "message": f"Only {len(valid_indices)} valid obs"})
            continue

        valid_values = col_values[valid_indices]

        # Vectorized returns calculation with suppressed warnings
        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(valid_values) / valid_values[:-1] * 100
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        if len(returns) < 100 or np.std(returns) < 1e-6:
            continue

        try:
            model = arch_model(returns, vol='Garch', p=1, q=1, dist='t', rescale=True, method_used = "analytical")
            result = model.fit(disp='off', show_warning=False,
                               options={'ftol': 1e-3, 'maxiter': 100})

            params = result.params
            alpha = params.get('alpha[1]', 0)
            beta = params.get('beta[1]', 0)
            nu = params.get('nu', 1000)

            if alpha + beta >= 1:
                logs.append({"ticker": ticker, "column": col, "error_type": "non_stationary",
                             "message": f"alpha+beta={alpha + beta:.4f} >= 1"})
                continue

            dynamic_threshold = t.ppf(1 - (confidence / 2), df=nu) if nu > 2 else 10.0

            cond_vol = result.conditional_volatility
            std_resid = result.resid / (cond_vol + 1e-8)

            outlier_mask_returns = np.abs(std_resid) > dynamic_threshold
            outlier_indices_returns = np.flatnonzero(outlier_mask_returns)

            if len(outlier_indices_returns) == 0:
                continue

            outlier_original_indices = valid_indices[outlier_indices_returns + 1]

            # Build clean array for interpolation
            clean_values_for_fit = col_values.copy()
            clean_values_for_fit[outlier_original_indices] = np.nan

            clean_valid_mask = np.isfinite(clean_values_for_fit)
            clean_valid_indices = np.flatnonzero(clean_valid_mask)
            clean_valid_values = clean_values_for_fit[clean_valid_indices]

            if len(clean_valid_indices) < 4:
                continue

            # Interpolation
            try:
                cs = CubicSpline(clean_valid_indices, clean_valid_values)
                corrected_values_array = cs(outlier_original_indices)
                method_used = "cubic_spline"
            except Exception:
                method_used = "fallback_nearest"
                idx_in_valid = np.searchsorted(clean_valid_indices, outlier_original_indices).clip(0,
                                                                                                   len(clean_valid_indices) - 1)
                corrected_values_array = clean_valid_values[idx_in_valid]

            # Handle non-finite corrected values
            non_finite_mask = ~np.isfinite(corrected_values_array)
            if np.any(non_finite_mask):
                for i in np.flatnonzero(non_finite_mask):
                    idx = outlier_original_indices[i]
                    nearest_idx = clean_valid_indices[np.abs(clean_valid_indices - idx).argmin()]
                    corrected_values_array[i] = clean_values_for_fit[nearest_idx]
                method_used = "last_valid_value_fallback"

            # Store original values before modification for logging
            original_values = col_values[outlier_original_indices].copy()

            # Batch update
            col_values[outlier_original_indices] = corrected_values_array
            modified_cols[col] = col_values

            # Vectorized logging (limited to MAX_CORRECTIONS_LOG)
            n_to_log = min(len(outlier_original_indices), MAX_CORRECTIONS_LOG)
            resid_vals = std_resid[outlier_indices_returns[:n_to_log]]

            logs.extend([
                {
                    "ticker": ticker,
                    "date": dates[outlier_original_indices[i]],
                    "column": col,
                    "error_type": "garch_outlier",
                    "original_value": float(original_values[i]),
                    "corrected_value": float(corrected_values_array[i]),
                    "standardized_residual": float(resid_vals[i]),
                    "threshold": dynamic_threshold,
                    "method": method_used
                }
                for i in range(n_to_log)
            ])

        except Exception as e:
            logs.append({"ticker": ticker, "column": col, "error_type": "garch_error", "message": str(e)})
            continue

    # Single DataFrame update at the end
    if modified_cols:
        working_df = working_df.with_columns([
            pl.Series(name=col, values=arr) for col, arr in modified_cols.items()
        ])

    # Simplified join using row indices
    result_lf = (
        df.lazy()
        .sort(date_col)
        .with_row_index("_join_idx")
        .drop(available_cols)
        .join(
            working_df.select(available_cols).lazy().with_row_index("_join_idx"),
            on="_join_idx",
            how="left"
        )
        .drop("_join_idx")
    )

    return result_lf if is_lazy else result_lf.collect(), logs