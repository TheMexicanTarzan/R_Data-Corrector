import polars as pl
import numpy as np
from typing import Union
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


def mahalanobis_filter(
        df: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.01
) -> tuple[Union[pl.DataFrame, pl.LazyFrame], list[dict]]:
    """
    Detect outliers using Pooled Robust Covariance (Highly Optimized).
    """
    # 1. Setup (Same as before)
    is_lazy = isinstance(df, pl.LazyFrame)
    working_lf = df if is_lazy else df.lazy()
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # Basic validations
    schema_cols = set(working_lf.collect_schema().names())
    if date_col not in schema_cols:
        return (working_lf.collect() if not is_lazy else working_lf, logs)  # Error log omitted for brevity

    available_cols = [col for col in columns if col in schema_cols]
    if len(available_cols) < 2:
        return (working_lf.collect() if not is_lazy else working_lf, logs)  # Error log omitted

    # 2. Peer Identification (Same as before)
    if metadata is None: return (working_lf.collect() if not is_lazy else working_lf, logs)

    try:
        meta_df = metadata.filter(pl.col("ticker") == ticker).select("sector").collect()
        if meta_df.is_empty() or meta_df["sector"][0] is None:
            return (working_lf.collect() if not is_lazy else working_lf, logs)
        target_sector = meta_df["sector"][0]
        peer_tickers = metadata.filter(pl.col("sector") == target_sector).select("ticker").collect()["ticker"].to_list()
    except Exception:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 3. Data Prep: Calculate Robust Z-Scores via Window Functions
    # Instead of iterating, we do all math in Polars engines

    # Calculate MAD constant: 1 / Q3(Standard Normal) approx 1.4826
    MAD_FACTOR = 1.4826

    # Define Expressions for Robust Standardization per Quarter
    z_score_exprs = []
    for col in available_cols:
        # Median per quarter
        median_expr = pl.col(col).median().over("_quarter")
        # MAD per quarter
        mad_expr = (pl.col(col) - median_expr).abs().median().over("_quarter") * MAD_FACTOR
        # Robust Z-score (add epsilon to avoid div by zero)
        z_expr = ((pl.col(col) - median_expr) / (mad_expr + 1e-8)).alias(f"_z_{col}")
        z_score_exprs.append(z_expr)

    # Materialize the Z-score matrix
    # Filter for peers, generate quarters, calculate Z-scores immediately
    subset_df = (
        working_lf
        .filter(pl.col("ticker").is_in(peer_tickers))
        .select(["ticker", date_col] + available_cols)
        .sort(date_col)
        .with_columns(
            (pl.col(date_col).dt.year().cast(pl.Utf8) + "-" +
             pl.col(date_col).dt.quarter().cast(pl.Utf8)).alias("_quarter")
        )
        .with_columns(z_score_exprs)  # <--- Calculation happens here in Rust
        .collect()
    )

    if subset_df.is_empty(): return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 4. Fit Global Robust Covariance (Pooled Approach)
    # We fit MinCovDet ONCE on the pooled Z-scores of the entire peer group history.

    z_cols = [f"_z_{c}" for c in available_cols]

    # Extract training data (all valid peers across all time)
    # Drop nulls so Sklearn doesn't complain
    training_matrix = subset_df.select(z_cols).drop_nulls().to_numpy()

    # Validation: Need enough data for the fit
    if len(training_matrix) < len(available_cols) * 5:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    try:
        # Fit ONCE
        mcd = MinCovDet(random_state=1)
        mcd.fit(training_matrix)

        # Get the Precision Matrix (Inverse Covariance)
        # We use this to calculate distance manually: D^2 = z * S^-1 * z.T
        robust_precision = mcd.precision_  # Shape (k, k)

        # We also get the location, though for Z-scores it should be near 0
        robust_location = mcd.location_  # Shape (k,)

    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "mcd_fit_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 5. Calculate Distances for Target Ticker (Vectorized)

    # Filter down to just our target ticker to score it
    target_df = subset_df.filter(pl.col("ticker") == ticker)
    if target_df.is_empty(): return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Extract target Z-scores
    target_z_matrix = target_df.select(z_cols).to_numpy()

    # Handle NaNs in target (can't score them)
    # Create mask of rows that are fully valid
    valid_mask = np.isfinite(target_z_matrix).all(axis=1)

    # Initialize distances array with NaNs
    distances_sq = np.full(len(target_df), np.nan)

    if np.any(valid_mask):
        # Vectorized Mahalanobis Calculation:
        # (X - mu)
        diff = target_z_matrix[valid_mask] - robust_location
        # (X - mu) @ S^-1
        left_term = diff @ robust_precision
        # ((X - mu) @ S^-1) * (X - mu) -> Sum over columns -> D^2
        # This is the diagonal of the full matrix multiplication
        dist_sq_valid = np.sum(left_term * diff, axis=1)

        distances_sq[valid_mask] = dist_sq_valid

    # 6. Detect & Correct
    chi2_threshold = chi2.ppf(1 - confidence, df=len(available_cols))

    # Identify violating indices (local to target_df)
    # Use np.greater with where to safely handle NaNs (NaN > thresh is False)
    violating_indices = np.where(np.greater(distances_sq, chi2_threshold, where=np.isfinite(distances_sq)))[0]

    violating_quarters = set()

    if len(violating_indices) > 0:
        quarters = target_df["_quarter"].to_list()
        dates = target_df[date_col].to_list()

        for idx in violating_indices:
            q = quarters[idx]
            dist = float(distances_sq[idx])

            # Add to set for vectorized imputation
            violating_quarters.add(q)

            if len(logs) < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],
                    "quarter": q,
                    "error_type": "mahalanobis_outlier",
                    "mahalanobis_distance_sq": dist,
                    "threshold": chi2_threshold
                })

    # 7. Imputation (Same Join-and-Switch Logic as before)
    if not violating_quarters:
        return (target_df.select(working_lf.columns).lazy() if is_lazy else target_df.select(working_lf.columns), logs)

    # Downsample target to quarterly for imputation source
    # We must operate on raw values, not Z-scores, for the fix
    target_quarterly_vals = (
        target_df
        .group_by("_quarter")
        .last()
        .sort("_quarter")
        .select(["_quarter"] + available_cols)
    )

    # Create Replacement Mapping (Vectorized Forward Fill)
    replacement_exprs = []
    for col in available_cols:
        expr = (
            pl.when(pl.col("_quarter").is_in(violating_quarters))
            .then(None)
            .otherwise(pl.col(col))
            .forward_fill()
            .shift(1)
            .alias(f"{col}_replacement")
        )
        replacement_exprs.append(expr)

    replacement_map = target_quarterly_vals.with_columns(replacement_exprs)

    # Join and Switch
    corrected_df = (
        target_df
        .join(replacement_map.select(["_quarter"] + [f"{c}_replacement" for c in available_cols]),
              on="_quarter",
              how="left")
    )

    final_cols = []
    for col in available_cols:
        cond = pl.col("_quarter").is_in(violating_quarters) & pl.col(f"{col}_replacement").is_not_null()
        final_cols.append(
            pl.when(cond)
            .then(pl.col(f"{col}_replacement"))
            .otherwise(pl.col(col))
            .alias(col)
        )

    result_df = corrected_df.with_columns(final_cols).select(working_lf.columns)

    if is_lazy:
        return result_df.lazy(), logs
    else:
        return result_df, logs