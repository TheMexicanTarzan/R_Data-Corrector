import polars as pl
import numpy as np
from numba import jit

@jit(nopython=True)
def add_double_mad_outliers(
        df: pl.DataFrame,
        columns: list[str],
        window_size: int = 30,
        threshold: float = 3.0
) -> pl.DataFrame:
    """
    Applies a Rolling Double MAD filter to flag outliers in specified columns.

    Args:
        df: Input Polars DataFrame.
        columns: List of column names to filter.
        window_size: Size of the rolling window.
        threshold: The Z-score threshold to flag outliers (default 3.0).

    Returns:
        DataFrame with new columns "{col}_outlier" (Boolean).
    """

    # 1. Define the specific Double MAD logic optimized for a 1D numpy array
    def _double_mad_window(arr: np.ndarray) -> bool:
        # Calculate the median of the current window
        median = np.median(arr)

        # Current value (last element in the rolling window)
        current_val = arr[-1]

        if current_val == median:
            return False

        # Split window into Left (<= median) and Right (>= median)
        # We perform absolute deviation calculation immediately
        abs_dev = np.abs(arr - median)

        left_mask = arr <= median
        right_mask = arr >= median

        # Calculate MAD for left and right tails
        # We add a tiny epsilon (1e-6) to prevent division by zero
        left_mad = np.median(abs_dev[left_mask])
        if left_mad == 0: left_mad = 1e-6

        right_mad = np.median(abs_dev[right_mask])
        if right_mad == 0: right_mad = 1e-6

        # Standard consistency constant for Gaussian distribution
        k = 1.4826

        # Calculate Double MAD Z-score based on which side the current value falls
        if current_val < median:
            score = abs(current_val - median) / (left_mad * k)
        else:
            score = abs(current_val - median) / (right_mad * k)

        return score > threshold

    # 2. Construct the expression list
    # We use pl.Boolean for memory optimality (1 bit/byte vs 64 bits for floats)
    exprs = [
        pl.col(col)
        .rolling(window_size=window_size)
        .map_elements(_double_mad_window, return_dtype=pl.Boolean)
        .alias(f"{col}_outlier")
        for col in columns
    ]

    # 3. Apply efficiently
    return df.with_columns(exprs)


import polars as pl
import numpy as np
import numba


@numba.jit(nopython=True)
def _run_adaptive_kalman(values, flags, r_normal, r_outlier, q_process):
    n = len(values)

    # State Vector: [position, velocity]
    # We initialize position at the first value, velocity at 0
    x_est = values[0]
    v_est = 0.0

    # Error Covariance Matrix (2x2)
    # P00 = pos_var, P01 = cov, P10 = cov, P11 = vel_var
    p00, p01, p10, p11 = 1.0, 0.0, 0.0, 1.0

    # Output array
    smoothed = np.empty(n)
    smoothed[0] = x_est

    # Kalman Loop
    for i in range(1, n):
        # 1. PREDICTION STEP (A * x)
        # Assuming constant velocity model (dt = 1)
        x_pred = x_est + v_est
        v_pred = v_est

        # Predict Covariance (A * P * A.T + Q)
        # Q is process noise (uncertainty in the model)
        p00_pred = p00 + p10 + p01 + p11 + q_process
        p01_pred = p01 + p11
        p10_pred = p10 + p11
        p11_pred = p11 + q_process

        # 2. DECIDE R (Measurement Noise)
        # If flagged, R is huge (ignore measurement). If valid, R is small.
        current_r = r_outlier if flags[i] else r_normal

        # 3. UPDATE STEP
        measurement = values[i]

        # Innovation (residual)
        y = measurement - x_pred

        # Innovation Covariance (S = H * P * H.T + R)
        # H is [1, 0] because we only measure position
        s = p00_pred + current_r

        # Kalman Gain (K = P * H.T * S^-1)
        k0 = p00_pred / s
        k1 = p10_pred / s

        # Update State
        x_est = x_pred + k0 * y
        v_est = v_pred + k1 * y

        # Update Covariance (P = (I - K * H) * P)
        # We manually expand matrix multiplication for speed
        p00 = p00_pred * (1 - k0)
        p01 = p01_pred * (1 - k0)
        p10 = -k1 * p00_pred + p10_pred
        p11 = -k1 * p01_pred + p11_pred

        smoothed[i] = x_est

    return smoothed


def smooth_flagged_data(
        df: pl.DataFrame,
        value_col: str,
        flag_col: str,
        R_normal: float = 0.1,
        R_outlier: float = 100000.0,
        Q: float = 0.01
) -> pl.DataFrame:
    """
    Applies an adaptive Kalman filter that ignores flagged outliers.

    Args:
        df: Polars DataFrame
        value_col: Column with raw data
        flag_col: Boolean column (True = Outlier)
        R_normal: Noise assumption for valid data (lower = follow data closer)
        R_outlier: Noise assumption for outliers (must be very high)
        Q: Process noise (allowance for true changes in velocity)
    """

    # Extract arrays
    vals = df[value_col].to_numpy().astype(np.float64)
    # Ensure flags are boolean (handles 1/0 or True/False)
    flags = df[flag_col].to_numpy().astype(bool)

    # Run optimized Numba filter
    smoothed_vals = _run_adaptive_kalman(vals, flags, R_normal, R_outlier, Q)

    return df.with_columns(
        pl.Series(f"{value_col}_smoothed", smoothed_vals)
    )