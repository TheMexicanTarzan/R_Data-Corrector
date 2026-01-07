import polars
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.stats import t
from numba import jit
import math


# ---------------------------------------------------------
# 1. Numba-Compiled Core with Student's t-Distribution
# ---------------------------------------------------------
@jit(nopython=True, cache=True)
def garch_recursion(params, returns, sigma2_0):
    """
    Computes GARCH(1,1) Log-Likelihood assuming Student's t-distribution.
    params: [omega, alpha, beta, nu]
    """
    omega = params[0]
    alpha = params[1]
    beta = params[2]
    nu = params[3]

    n = len(returns)
    sigma2 = np.zeros(n)
    sigma2[0] = sigma2_0

    log_lik = 0.0

    # Pre-calculate constant terms for the t-distribution LL
    # This avoids re-calculating Gamma functions inside the loop
    # Formula uses Log-Gamma (lgamma) for numerical stability
    term1 = math.lgamma((nu + 1) / 2)
    term2 = math.lgamma(nu / 2)
    term3 = 0.5 * math.log(math.pi * (nu - 2))
    const_part = term1 - term2 - term3

    for t in range(1, n):
        # 1. Variance Recursion (Same as Normal GARCH)
        # sigma^2_t = omega + alpha * r^2_{t-1} + beta * sigma^2_{t-1}
        prev_ret_sq = returns[t - 1] ** 2
        sigma2[t] = omega + alpha * prev_ret_sq + beta * sigma2[t - 1]

        # 2. Student's t Log-Likelihood
        # LL_t = const - 0.5*log(sigma^2) - ((nu+1)/2) * log(1 + r^2 / (sigma^2 * (nu-2)))
        curr_var = sigma2[t]

        # Avoid log(0) or division by zero
        if curr_var > 1e-9:
            std_resid_sq = (returns[t] ** 2) / (curr_var * (nu - 2))
            log_term = math.log(1 + std_resid_sq)

            # Accumulate Negative Log Likelihood
            ll_t = const_part - 0.5 * math.log(curr_var) - ((nu + 1) / 2) * log_term
            log_lik += ll_t
        else:
            log_lik -= 1e10  # Heavy penalty for impossible variance

    # Return Negative Log-Likelihood (for minimization)
    return -log_lik, sigma2


def fit_garch_numba(returns):
    """
    Fits GARCH(1,1) with Student's t-distribution.
    Returns: (omega, alpha, beta, nu, conditional_volatility)
    """
    var_target = np.var(returns)
    if var_target == 0:
        return 0, 0, 0, 1000, np.zeros_like(returns)

    sigma2_0 = var_target

    # Initial Params: [omega, alpha, beta, nu]
    # Guess nu = 8.0 (fairly fat tails)
    initial_params = [0.01 * var_target, 0.05, 0.90, 8.0]

    # Bounds:
    # omega > 0
    # 0 <= alpha < 1
    # 0 <= beta < 1
    # 2.01 <= nu <= 100 (Variance undefined if nu <= 2)
    bounds = ((1e-6, None), (1e-6, 1.0), (1e-6, 1.0), (2.01, 100.0))

    def objective(params):
        # Stationarity: alpha + beta < 1
        if params[1] + params[2] >= 0.999:
            return 1e10
        nll, _ = garch_t_recursion(params, returns, sigma2_0)
        return nll

    # Run Optimization
    # We relax ftol slightly (1e-3) for speed, but t-dist is harder to fit so we allow more iters
    res = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B', tol=1e-3)

    # Extract results
    omega, alpha, beta, nu = res.x

    # Re-run recursion to get final volatility series
    _, best_sigma2 = garch_t_recursion(res.x, returns, sigma2_0)

    return omega, alpha, beta, nu, np.sqrt(best_sigma2)

# ---------------------------------------------------------
# 2. Main Filter Function
# ---------------------------------------------------------
def garch_residuals(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.001,
        shared_data: dict = None  # Unused - for interface consistency with cross-sectional filters
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using GARCH(1,1) (Numba Optimized).
    """
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

    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()
    dates = working_df[date_col].to_list()

    # Explicit copy for writable memory
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

        # Safe Returns Calculation
        denom = valid_values[:-1]
        denom_safe = np.where(denom == 0, np.nan, denom)

        with np.errstate(divide='ignore', invalid='ignore'):
            returns = np.diff(valid_values) / denom_safe * 100

        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

        # Skip low variance or insufficient data
        if len(returns) < 100 or np.std(returns) < 1e-6:
            continue

        try:
            # === Fast Numba Fit ===
            # Ensure contiguous float64 array for Numba
            returns_c = np.ascontiguousarray(returns, dtype=np.float64)

            omega, alpha, beta, nu, cond_vol = fit_garch_numba(returns_c)

            if alpha + beta >= 1:
                logs.append({"ticker": ticker, "column": col, "error_type": "non_stationary",
                             "message": f"alpha+beta={alpha + beta:.4f} >= 1"})
                continue

            # Standardized Residuals
            # cond_vol matches length of returns
            # Add epsilon to prevent div by zero
            std_resid = returns / (cond_vol + 1e-8)

            threshold = t.ppf(1 - (confidence / 2), df=nu)

            outlier_mask_returns = np.abs(std_resid) > threshold
            outlier_indices_returns = np.flatnonzero(outlier_mask_returns)

            if len(outlier_indices_returns) == 0:
                continue

            # Map back to price indices (shift + 1)
            outlier_original_indices = valid_indices[outlier_indices_returns + 1]

            # Interpolation Preparation
            clean_values_for_fit = col_values.copy()
            clean_values_for_fit[outlier_original_indices] = np.nan

            clean_valid_mask = np.isfinite(clean_values_for_fit)
            clean_valid_indices = np.flatnonzero(clean_valid_mask)
            clean_valid_values = clean_values_for_fit[clean_valid_indices]

            if len(clean_valid_indices) < 4:
                continue

            # Spline Interpolation
            try:
                cs = CubicSpline(clean_valid_indices, clean_valid_values)
                corrected_values_array = cs(outlier_original_indices)
                method_used = "cubic_spline"
            except Exception:
                method_used = "fallback_nearest"
                idx_in_valid = np.searchsorted(clean_valid_indices, outlier_original_indices).clip(0,
                                                                                                   len(clean_valid_indices) - 1)
                corrected_values_array = clean_valid_values[idx_in_valid]

            # Post-interpolation validation
            non_finite_mask = ~np.isfinite(corrected_values_array)
            if np.any(non_finite_mask):
                for i in np.flatnonzero(non_finite_mask):
                    idx = outlier_original_indices[i]
                    nearest_idx = clean_valid_indices[np.abs(clean_valid_indices - idx).argmin()]
                    corrected_values_array[i] = clean_values_for_fit[nearest_idx]
                method_used = "last_valid_value_fallback"

            # Log Preparation
            original_values = col_values[outlier_original_indices].copy()

            # Apply Updates
            col_values[outlier_original_indices] = corrected_values_array
            modified_cols[col] = col_values

            # Batch Logging
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
                    "threshold": threshold,
                    "method": method_used
                }
                for i in range(n_to_log)
            ])

        except Exception as e:
            logs.append({"ticker": ticker, "column": col, "error_type": "garch_error", "message": str(e)})
            continue

    # Final DataFrame Construction
    if modified_cols:
        working_df = working_df.with_columns([
            polars.Series(name=col, values=arr) for col, arr in modified_cols.items()
        ])

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