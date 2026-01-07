import polars
import numpy
from typing import Union
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


def mahalanobis_filter(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.01,
        shared_data: dict = None  # Required for cross-sectional peer analysis
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect outliers using Pooled Robust Covariance (Highly Optimized).

    This filter performs cross-sectional analysis by comparing the target ticker
    against its sector peers. It requires:
    - metadata: Full metadata LazyFrame to identify peer tickers in the same sector
    - shared_data: Dict mapping ticker symbols to (LazyFrame, metadata) tuples
                   for accessing peer ticker data
    """
    # 1. Setup
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # Clean ticker name (remove .csv extension if present)
    ticker_symbol = ticker.replace(".csv", "") if ticker.endswith(".csv") else ticker

    # Basic validations
    schema_cols = set(working_lf.collect_schema().names())
    if date_col not in schema_cols:
        logs.append({"ticker": ticker, "error_type": "missing_date_col",
                     "message": f"Date column '{date_col}' not found"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    available_cols = [col for col in columns if col in schema_cols]
    if len(available_cols) < 2:
        logs.append({"ticker": ticker, "error_type": "insufficient_cols",
                     "message": f"Need at least 2 columns, found {len(available_cols)}"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 2. Peer Identification - requires both metadata and shared_data
    if metadata is None:
        logs.append({"ticker": ticker, "error_type": "missing_metadata",
                     "message": "Metadata is required for cross-sectional analysis"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    if shared_data is None:
        logs.append({"ticker": ticker, "error_type": "missing_shared_data",
                     "message": "shared_data is required for peer comparison"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    try:
        # Use ticker_symbol (without .csv) to match metadata's "symbol" column
        meta_df = metadata.filter(polars.col("symbol") == ticker_symbol).select("sector").collect()
        if meta_df.is_empty() or meta_df["sector"][0] is None:
            logs.append({"ticker": ticker, "error_type": "no_sector",
                         "message": f"Ticker '{ticker_symbol}' not found in metadata or has no sector"})
            return (working_lf.collect() if not is_lazy else working_lf, logs)
        target_sector = meta_df["sector"][0]

        # Get all peer ticker symbols in the same sector
        peer_symbols = metadata.filter(polars.col("sector") == target_sector).select("symbol").collect()["symbol"].to_list()
    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "metadata_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 3. Collect Peer Data from shared_data
    # shared_data maps filename (e.g., "AAPL.csv") to (LazyFrame, metadata) tuples
    peer_frames = []
    for filename, payload in shared_data.items():
        # Extract ticker symbol from filename (e.g., "AAPL.csv" -> "AAPL")
        symbol = filename.replace(".csv", "")
        if symbol in peer_symbols:
            # Extract the LazyFrame from the tuple
            if isinstance(payload, tuple):
                peer_lf = payload[0]
            else:
                peer_lf = payload

            # Select only needed columns and add ticker identifier
            try:
                peer_schema = set(peer_lf.collect_schema().names())
                cols_to_select = [c for c in [date_col] + available_cols if c in peer_schema]
                if len(cols_to_select) > 1:  # At least date + 1 data column
                    peer_df = (
                        peer_lf
                        .select(cols_to_select)
                        .with_columns(polars.lit(symbol).alias("ticker"))
                        .collect()
                    )
                    peer_frames.append(peer_df)
            except Exception:
                continue  # Skip problematic tickers

    if len(peer_frames) < 3:
        logs.append({"ticker": ticker, "error_type": "insufficient_peers",
                     "message": f"Only {len(peer_frames)} valid peers found, need at least 3"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Concatenate all peer data
    pooled_df = polars.concat(peer_frames, how="diagonal")

    # 4. Data Prep: Calculate Robust Z-Scores via Window Functions
    MAD_FACTOR = 1.4826

    # Define Expressions for Robust Standardization per Quarter
    z_score_exprs = []
    for col in available_cols:
        if col not in pooled_df.columns:
            continue
        # Median per quarter
        median_expr = polars.col(col).median().over("_quarter")
        # MAD per quarter
        mad_expr = (polars.col(col) - median_expr).abs().median().over("_quarter") * MAD_FACTOR
        # Robust Z-score (add epsilon to avoid div by zero)
        z_expr = ((polars.col(col) - median_expr) / (mad_expr + 1e-8)).alias(f"_z_{col}")
        z_score_exprs.append(z_expr)

    # Update available_cols to only those present in pooled data
    available_cols = [col for col in available_cols if col in pooled_df.columns]
    if len(available_cols) < 2:
        logs.append({"ticker": ticker, "error_type": "insufficient_cols_after_pool",
                     "message": "Not enough common columns across peers"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Generate quarter column and calculate Z-scores
    subset_df = (
        pooled_df
        .sort(date_col)
        .with_columns(
            (polars.col(date_col).dt.year().cast(polars.Utf8) + "-" +
             polars.col(date_col).dt.quarter().cast(polars.Utf8)).alias("_quarter")
        )
        .with_columns(z_score_exprs)
    )

    if subset_df.is_empty():
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 5. Fit Global Robust Covariance (Pooled Approach)
    z_cols = [f"_z_{c}" for c in available_cols]

    # Extract training data (all valid peers across all time)
    training_matrix = subset_df.select(z_cols).drop_nulls().to_numpy()

    # Validation: Need enough data for the fit
    if len(training_matrix) < len(available_cols) * 5:
        logs.append({"ticker": ticker, "error_type": "insufficient_training_data",
                     "message": f"Only {len(training_matrix)} rows, need {len(available_cols) * 5}"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    try:
        # Fit ONCE on all peer data
        mcd = MinCovDet(random_state=1)
        mcd.fit(training_matrix)
        robust_precision = mcd.precision_
        robust_location = mcd.location_

    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "mcd_fit_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # 6. Calculate Distances for Target Ticker (Vectorized)
    target_df = subset_df.filter(polars.col("ticker") == ticker_symbol)
    if target_df.is_empty():
        logs.append({"ticker": ticker, "error_type": "no_target_data",
                     "message": f"Target ticker '{ticker_symbol}' not found in pooled data"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Extract target Z-scores
    target_z_matrix = target_df.select(z_cols).to_numpy()

    # Handle NaNs in target (can't score them)
    valid_mask = numpy.isfinite(target_z_matrix).all(axis=1)

    # Initialize distances array with NaNs
    distances_sq = numpy.full(len(target_df), numpy.nan)

    if numpy.any(valid_mask):
        # Vectorized Mahalanobis Calculation
        diff = target_z_matrix[valid_mask] - robust_location
        left_term = diff @ robust_precision
        dist_sq_valid = numpy.sum(left_term * diff, axis=1)
        distances_sq[valid_mask] = dist_sq_valid

    # 7. Detect & Correct
    chi2_threshold = chi2.ppf(1 - confidence, df=len(available_cols))

    # Identify violating indices
    violating_indices = numpy.where(
        numpy.greater(distances_sq, chi2_threshold, where=numpy.isfinite(distances_sq))
    )[0]

    violating_quarters = set()

    if len(violating_indices) > 0:
        quarters = target_df["_quarter"].to_list()
        dates = target_df[date_col].to_list()

        for idx in violating_indices:
            q = quarters[idx]
            dist = float(distances_sq[idx])

            violating_quarters.add(q)

            if len(logs) < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],
                    "quarter": q,
                    "error_type": "mahalanobis_outlier",
                    "mahalanobis_distance_sq": dist,
                    "threshold": chi2_threshold,
                    "num_peers": len(peer_frames)
                })

    # 8. Return Results
    # Get the original columns from working_lf
    original_cols = working_lf.collect_schema().names()

    if not violating_quarters:
        # No corrections needed - return original data
        if is_lazy:
            return working_lf, logs
        else:
            return working_lf.collect(), logs

    # 9. Imputation for violating quarters
    # Downsample target to quarterly for imputation source
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
            polars.when(polars.col("_quarter").is_in(violating_quarters))
            .then(None)
            .otherwise(polars.col(col))
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
        cond = polars.col("_quarter").is_in(violating_quarters) & polars.col(f"{col}_replacement").is_not_null()
        final_cols.append(
            polars.when(cond)
            .then(polars.col(f"{col}_replacement"))
            .otherwise(polars.col(col))
            .alias(col)
        )

    # Select only original columns that exist in corrected_df
    result_cols = [c for c in original_cols if c in corrected_df.columns or c in [col for col in available_cols]]
    result_df = corrected_df.with_columns(final_cols).select(
        [c for c in original_cols if c in corrected_df.with_columns(final_cols).columns]
    )

    if is_lazy:
        return result_df.lazy(), logs
    else:
        return result_df, logs
