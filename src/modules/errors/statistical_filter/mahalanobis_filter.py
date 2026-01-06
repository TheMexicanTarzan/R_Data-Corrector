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
    confidence: float = 0.01
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct outliers using Cross-Sectional Multivariate Outlier Detection.

    This filter uses Mahalanobis distance with a robust covariance estimator (MinCovDet)
    to identify multivariate outliers in quarterly fundamental data. The analysis is
    performed cross-sectionally within the same sector peer group.

    IMPORTANT: This is a cross-sectional filter. The input df MUST contain data for
    multiple tickers (the entire universe or at least all sector peers). If df only
    contains the target ticker's data, the function will return it unchanged with
    an appropriate log message.

    Process:
        1. Use metadata to identify the target ticker's sector
        2. Filter df to include all tickers sharing the same sector
        3. Downsample daily data to quarterly (last valid entry per quarter)
        4. Fit MinCovDet to estimate robust covariance and mean
        5. Calculate Mahalanobis Distance D²_M for the target ticker's quarterly points
        6. Flag quarters where D²_M > χ²(p, 0.99)
        7. Broadcast flags to all daily entries within violating quarters
        8. Forward-fill entire violating quarter with last valid observation from previous quarter

    Args:
        df: Input DataFrame or LazyFrame containing daily data for the universe of
            tickers. Must contain a "ticker" column to identify different securities.
            For cross-sectional analysis to work, df must include data for peer tickers.
        metadata: LazyFrame containing metadata with at least "ticker" and "sector" columns.
            Used to identify which tickers belong to the same sector.
        ticker: Target ticker symbol to analyze and correct.
        columns: List of column names (fundamentals) to include in multivariate analysis.
        date_col: Name of the date column (default: 'm_date').
        confidence: Confidence level for chi-square threshold (default: 0.01 for 99%).

    Returns:
        tuple containing:
            - Corrected DataFrame/LazyFrame containing ONLY the target ticker's data
            - List of dictionaries documenting each correction made
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())

    # Initialize logs
    logs = []
    MAX_CORRECTIONS_LOG = 50

    # Validate required columns
    if date_col not in schema_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "missing_date_column",
            "message": f"Date column '{date_col}' not found in dataframe"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker) if "ticker" in schema_cols else working_lf
        return (result_lf if is_lazy else result_lf.collect(), logs)

    if "ticker" not in schema_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "missing_ticker_column",
            "message": "Ticker column not found in dataframe"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Check metadata for sector information
    if metadata is None:
        logs.append({
            "ticker": ticker,
            "error_type": "missing_metadata",
            "message": "Metadata not provided, cannot determine sector peer group"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    metadata_schema = metadata.collect_schema()
    if "sector" not in metadata_schema.names():
        logs.append({
            "ticker": ticker,
            "error_type": "missing_sector_column",
            "message": "Sector column not found in metadata"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Filter available analysis columns
    available_cols = [col for col in columns if col in schema_cols]

    if len(available_cols) < 2:
        logs.append({
            "ticker": ticker,
            "error_type": "insufficient_columns",
            "message": f"Need at least 2 columns for multivariate analysis, found {len(available_cols)}"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Get target ticker's sector from metadata
    target_sector_df = (
        metadata
        .filter(polars.col("ticker") == ticker)
        .select("sector")
        .limit(1)
        .collect()
    )

    if target_sector_df.is_empty():
        logs.append({
            "ticker": ticker,
            "error_type": "ticker_not_in_metadata",
            "message": f"Ticker '{ticker}' not found in metadata"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    target_sector = target_sector_df["sector"][0]

    if target_sector is None:
        logs.append({
            "ticker": ticker,
            "error_type": "null_sector",
            "message": f"Sector is null for ticker '{ticker}'"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Get all tickers in the same sector from metadata
    peer_tickers_df = (
        metadata
        .filter(polars.col("sector") == target_sector)
        .select("ticker")
        .collect()
    )

    peer_tickers = peer_tickers_df["ticker"].to_list()

    if len(peer_tickers) < 5:
        logs.append({
            "ticker": ticker,
            "error_type": "insufficient_peers",
            "message": f"Only {len(peer_tickers)} peers in sector '{target_sector}', need at least 5"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Check which peer tickers actually exist in df
    tickers_in_df = (
        working_lf
        .select("ticker")
        .unique()
        .collect()
    )["ticker"].to_list()

    available_peers = [t for t in peer_tickers if t in tickers_in_df]

    if len(available_peers) < 5:
        logs.append({
            "ticker": ticker,
            "error_type": "insufficient_peer_data",
            "message": f"Only {len(available_peers)} peers found in df (need 5). "
                       f"Ensure df contains data for the full universe or sector peers."
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    if ticker not in tickers_in_df:
        logs.append({
            "ticker": ticker,
            "error_type": "target_ticker_missing",
            "message": f"Target ticker '{ticker}' not found in dataframe"
        })
        # Return empty LazyFrame with correct schema
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Filter df to include only peer tickers (including target)
    select_cols = ["ticker", date_col] + available_cols
    peer_df = (
        working_lf
        .filter(polars.col("ticker").is_in(available_peers))
        .select(select_cols)
        .collect()
    )

    if peer_df.is_empty():
        logs.append({
            "ticker": ticker,
            "error_type": "no_peer_data",
            "message": "No data found for peer tickers"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Add quarter column for downsampling
    # Quarter format: YYYY-Q
    peer_df = peer_df.with_columns(
        (
            polars.col(date_col).dt.year().cast(polars.Utf8) +
            polars.lit("-Q") +
            polars.col(date_col).dt.quarter().cast(polars.Utf8)
        ).alias("_quarter")
    )

    # Downsample to quarterly: last valid entry per (ticker, quarter)
    # First sort by date, then take last row per group
    quarterly_df = (
        peer_df
        .sort(date_col)
        .group_by(["ticker", "_quarter"])
        .last()
    )

    # Get unique quarters for analysis
    unique_quarters = quarterly_df["_quarter"].unique().sort().to_list()

    if len(unique_quarters) < 4:
        logs.append({
            "ticker": ticker,
            "error_type": "insufficient_quarters",
            "message": f"Only {len(unique_quarters)} quarters available, need at least 4"
        })
        # Return only target ticker's data
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    # Chi-square threshold at specified confidence level
    p = len(available_cols)  # Number of dimensions
    chi2_threshold = chi2.ppf(1 - confidence, df=p)

    # Track violating quarters for the target ticker
    violating_quarters = []
    col_corrections_logged = 0

    # Analyze each quarter cross-sectionally
    for quarter in unique_quarters:
        # Get cross-sectional data for this quarter
        quarter_data = quarterly_df.filter(polars.col("_quarter") == quarter)

        # Extract numeric data for MCD
        data_matrix = quarter_data.select(available_cols).to_numpy()

        # Remove rows with any NaN
        valid_mask = ~numpy.isnan(data_matrix).any(axis=1)
        valid_data = data_matrix[valid_mask]
        valid_tickers = quarter_data.filter(
            polars.Series(valid_mask)
        )["ticker"].to_list()

        if len(valid_data) < len(available_cols) + 2:
            # Not enough observations for MCD estimation
            continue

        # Check if target ticker is in this quarter's data
        if ticker not in valid_tickers:
            continue

        target_idx = valid_tickers.index(ticker)
        target_observation = valid_data[target_idx].reshape(1, -1)

        try:
            # Fit MinCovDet for robust estimation
            mcd = MinCovDet(random_state=42)
            mcd.fit(valid_data)

            # Get robust location and covariance
            robust_mean = mcd.location_
            robust_cov = mcd.covariance_

            # Calculate Mahalanobis distance for target ticker
            # D² = (x - μ)ᵀ Σ⁻¹ (x - μ)
            diff = target_observation - robust_mean
            inv_cov = numpy.linalg.inv(robust_cov)
            mahal_dist_sq = float(diff @ inv_cov @ diff.T)

            # Check against chi-square threshold
            if mahal_dist_sq > chi2_threshold:
                violating_quarters.append({
                    "quarter": quarter,
                    "mahalanobis_distance_sq": mahal_dist_sq,
                    "threshold": chi2_threshold,
                    "observation": {col: float(target_observation[0, i])
                                    for i, col in enumerate(available_cols)}
                })

                if col_corrections_logged < MAX_CORRECTIONS_LOG:
                    logs.append({
                        "ticker": ticker,
                        "quarter": quarter,
                        "error_type": "mahalanobis_outlier",
                        "mahalanobis_distance_sq": mahal_dist_sq,
                        "chi2_threshold": chi2_threshold,
                        "dimensions": p,
                        "confidence_level": 1 - confidence,
                        "num_peers": len(valid_tickers),
                        "sector": target_sector
                    })
                    col_corrections_logged += 1

        except Exception as exc:
            logs.append({
                "ticker": ticker,
                "quarter": quarter,
                "error_type": "mcd_fit_error",
                "message": str(exc)
            })
            continue

    # Extract target ticker's daily data for returning
    target_daily_df = (
        working_lf
        .filter(polars.col("ticker") == ticker)
        .sort(date_col)
        .collect()
    )

    if target_daily_df.is_empty():
        result_lf = working_lf.filter(polars.col("ticker") == ticker)
        return (result_lf if is_lazy else result_lf.collect(), logs)

    if not violating_quarters:
        # No outliers found, return target ticker's data unchanged
        if is_lazy:
            return target_daily_df.lazy(), logs
        else:
            return target_daily_df, logs

    # Impute the violating quarters for the target ticker
    # Add quarter column to target daily data
    target_daily_df = target_daily_df.with_columns(
        (
            polars.col(date_col).dt.year().cast(polars.Utf8) +
            polars.lit("-Q") +
            polars.col(date_col).dt.quarter().cast(polars.Utf8)
        ).alias("_quarter")
    )

    # Get violating quarter strings
    violating_quarter_strs = [vq["quarter"] for vq in violating_quarters]

    # For each violating quarter, find the last valid observation from previous quarter
    dates = target_daily_df[date_col].to_list()
    quarters = target_daily_df["_quarter"].to_list()

    # Create numpy arrays for each column to modify
    column_arrays = {}
    for col in available_cols:
        column_arrays[col] = target_daily_df[col].to_numpy().astype(numpy.float64)

    # Sort quarters to find previous valid quarter
    sorted_quarters = sorted(set(quarters))
    quarter_to_prev_valid = {}

    for i, q in enumerate(sorted_quarters):
        if q in violating_quarter_strs:
            # Find the most recent non-violating quarter before this one
            prev_valid_quarter = None
            for j in range(i - 1, -1, -1):
                if sorted_quarters[j] not in violating_quarter_strs:
                    prev_valid_quarter = sorted_quarters[j]
                    break
            quarter_to_prev_valid[q] = prev_valid_quarter

    # For each violating quarter, get last observation from previous valid quarter
    for vq in violating_quarter_strs:
        prev_quarter = quarter_to_prev_valid.get(vq)

        if prev_quarter is None:
            # No previous valid quarter available
            logs.append({
                "ticker": ticker,
                "quarter": vq,
                "error_type": "no_previous_valid_quarter",
                "message": "Cannot impute: no valid quarter before violating quarter"
            })
            continue

        # Find indices for violating quarter
        violating_indices = [i for i, q in enumerate(quarters) if q == vq]

        # Find the last observation from previous valid quarter
        prev_quarter_indices = [i for i, q in enumerate(quarters) if q == prev_quarter]

        if not prev_quarter_indices:
            continue

        last_valid_idx = prev_quarter_indices[-1]

        # Forward-fill: replace all values in violating quarter with last valid value
        for col in available_cols:
            last_valid_value = column_arrays[col][last_valid_idx]

            if not numpy.isfinite(last_valid_value):
                # Last value of previous quarter is null, try to find any valid value
                for idx in reversed(prev_quarter_indices):
                    if numpy.isfinite(column_arrays[col][idx]):
                        last_valid_value = column_arrays[col][idx]
                        break

            if numpy.isfinite(last_valid_value):
                for idx in violating_indices:
                    original_value = column_arrays[col][idx]
                    column_arrays[col][idx] = last_valid_value

                    if col_corrections_logged < MAX_CORRECTIONS_LOG and idx == violating_indices[0]:
                        logs.append({
                            "ticker": ticker,
                            "date": dates[idx],
                            "quarter": vq,
                            "column": col,
                            "error_type": "mahalanobis_imputation",
                            "original_value": float(original_value) if numpy.isfinite(original_value) else None,
                            "corrected_value": float(last_valid_value),
                            "source_quarter": prev_quarter,
                            "method": "forward_fill_quarter"
                        })
                        col_corrections_logged += 1

    # Update target daily dataframe with corrected columns
    for col in available_cols:
        target_daily_df = target_daily_df.with_columns(
            polars.Series(name=col, values=column_arrays[col])
        )

    # Remove the _quarter helper column
    target_daily_df = target_daily_df.drop("_quarter")

    # Return only the target ticker's corrected data
    if is_lazy:
        return target_daily_df.lazy(), logs
    else:
        return target_daily_df, logs
