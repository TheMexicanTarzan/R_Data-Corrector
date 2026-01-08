import polars as pl
import numpy as np
import scipy.linalg
from typing import Union
from sklearn.covariance import MinCovDet
from scipy.stats import chi2


def mahalanobis_filter(
        df: Union[pl.DataFrame, pl.LazyFrame],
        metadata: pl.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.0001,
        shared_data: dict = None
) -> tuple[Union[pl.DataFrame, pl.LazyFrame], list[dict]]:
    """
    Detect outliers using 'Sector Cloud' approach:
    1. Collapses Daily -> Quarterly (Fixes Singularity)
    2. Normalizes per Quarter (Fixes Time Alignment)
    3. Pools all history to learn Sector Geometry
    """
    # --- 1. Setup & Validation ---
    is_lazy = isinstance(df, pl.LazyFrame)
    working_lf = df if is_lazy else df.lazy()
    logs = []
    MAX_CORRECTIONS_LOG = 50

    ticker_symbol = ticker.replace(".csv", "") if ticker.endswith(".csv") else ticker

    # Schema check
    schema_cols = set(working_lf.collect_schema().names())
    if date_col not in schema_cols:
        logs.append({"ticker": ticker, "error_type": "missing_date", "message": "Date col missing"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    available_cols = [col for col in columns if col in schema_cols]
    if len(available_cols) < 2:
        logs.append({"ticker": ticker, "error_type": "insufficient_cols", "message": "<2 cols"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    if shared_data is None:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # --- 2. Peer Identification ---
    try:
        # Get Target Sector
        meta_df = metadata.filter(pl.col("symbol") == ticker_symbol).select("sector").collect()
        if meta_df.is_empty() or meta_df["sector"][0] is None:
            return (working_lf.collect() if not is_lazy else working_lf, logs)

        target_sector = meta_df["sector"][0]
        peer_symbols = metadata.filter(pl.col("sector") == target_sector).select("symbol").collect()["symbol"].to_list()
    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "meta_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # --- 3. Collect & Collapse Data (Daily -> Quarterly) ---
    # We process peers and target identically: Collapse to 1 row per quarter
    quarterly_frames = []

    # 3a. Process Target (from input df)
    # We add a temp column _quarter_id (e.g., "2020-1")
    target_q_lf = (
        working_lf
        .select([date_col] + available_cols)
        .with_columns([
            pl.lit(ticker_symbol).alias("ticker"),
            (pl.col(date_col).dt.year().cast(pl.Utf8) + "-" +
             pl.col(date_col).dt.quarter().cast(pl.Utf8)).alias("_quarter_id")
        ])
        .group_by(["ticker", "_quarter_id"])
        .agg([
            pl.col(date_col).last(),  # Keep last date of quarter for reference
            *[pl.col(c).last() for c in available_cols]  # Keep last fundamental value
        ])
    )
    quarterly_frames.append(target_q_lf.collect())

    # 3b. Process Peers (from shared_data)
    for filename, payload in shared_data.items():
        symbol = filename.replace(".csv", "")
        if symbol in peer_symbols and symbol != ticker_symbol:
            peer_lf = payload[0] if isinstance(payload, tuple) else payload
            try:
                peer_schema = set(peer_lf.collect_schema().names())
                cols_curr = [c for c in [date_col] + available_cols if c in peer_schema]

                # Must have date + at least 1 feature
                if len(cols_curr) > 1:
                    q_peer = (
                        peer_lf
                        .select(cols_curr)
                        .with_columns([
                            pl.lit(symbol).alias("ticker"),
                            (pl.col(date_col).dt.year().cast(pl.Utf8) + "-" +
                             pl.col(date_col).dt.quarter().cast(pl.Utf8)).alias("_quarter_id")
                        ])
                        .group_by(["ticker", "_quarter_id"])
                        .agg([
                            pl.col(date_col).last(),
                            *[pl.col(c).last() for c in cols_curr if c != date_col]
                        ])
                        .collect()
                    )
                    quarterly_frames.append(q_peer)
            except Exception:
                continue

    if len(quarterly_frames) < 2:  # Target + at least 1 peer
        logs.append({"ticker": ticker, "error_type": "no_peers", "message": "No peer data found"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Combine all quarterly snapshots
    pooled_q_df = pl.concat(quarterly_frames, how="diagonal")

    # --- 4. Time-Aware Normalization (The "Ragged Edge" Fix) ---
    # We calculate Z-scores grouped by QUARTER.
    # This aligns 2010 data with 2020 data by removing the time-trend component.

    # Filter only cols that exist in pooled data
    valid_cols = [c for c in available_cols if c in pooled_q_df.columns]

    z_exprs = []
    MAD_FACTOR = 1.4826

    for col in valid_cols:
        # Calculate Median per specific quarter (e.g., median of all peers in Q1 2020)
        median_expr = pl.col(col).median().over("_quarter_id")

        # Calculate MAD per specific quarter
        mad_expr = (pl.col(col) - median_expr).abs().median().over("_quarter_id") * MAD_FACTOR

        # Safe Scale: If MAD is 0 (all peers have same value), use fallback
        # This handles cases where only 1 or 2 peers exist in a specific quarter
        safe_scale = (
            pl.when(mad_expr < 1e-6)
            .then(pl.col(col).abs().mean().over("_quarter_id") * 0.01 + 1e-6)
            .otherwise(mad_expr)
        )

        z_exprs.append(
            ((pl.col(col) - median_expr) / safe_scale).alias(f"_z_{col}")
        )

    # Apply Normalization
    # Filter: We only keep quarters where we have at least 3 tickers to form a valid "Cluster"
    # Otherwise, Z-scores are meaningless (e.g. comparing ticker against itself)
    normalized_df = (
        pooled_q_df
        .with_columns(pl.col("ticker").count().over("_quarter_id").alias("_peer_count"))
        .filter(pl.col("_peer_count") >= 3)
        .with_columns(z_exprs)
    )

    if normalized_df.filter(pl.col("ticker") == ticker_symbol).is_empty():
        logs.append({"ticker": ticker, "error_type": "insufficient_history", "message": "Not enough overlapping peers"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # --- 5. Train "Sector Cloud" (MCD) ---
    z_cols = [f"_z_{c}" for c in valid_cols]

    # We train on EVERYONE (Peers + Target) across ALL TIME.
    # This creates the "Sector Definition"
    training_matrix = normalized_df.select(z_cols).drop_nulls().to_numpy()

    # Safety: Need enough total points
    if len(training_matrix) < len(valid_cols) * 5:
        logs.append({"ticker": ticker, "error_type": "training_fail", "message": "N < 5p"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    try:
        # We can use high support because we have cleaned the data structure
        mcd = MinCovDet(random_state=42, support_fraction=0.9, assume_centered=False)
        mcd.fit(training_matrix)

        robust_cov = mcd.covariance_
        robust_loc = mcd.location_

        # Regularize for safety (invertibility)
        robust_cov.flat[::robust_cov.shape[0] + 1] += 1e-6
        robust_prec = scipy.linalg.pinvh(robust_cov)

    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "mcd_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # --- 6. Score Target Quarters ---
    target_z_df = normalized_df.filter(pl.col("ticker") == ticker_symbol)
    target_mat = target_z_df.select(z_cols).to_numpy()

    # Calculate Distances
    distances = np.full(len(target_z_df), np.nan)
    mask = np.isfinite(target_mat).all(axis=1)

    if np.any(mask):
        diff = target_mat[mask] - robust_loc
        # (x-u)P(x-u)'
        distances[mask] = np.sum((diff @ robust_prec) * diff, axis=1)

    # --- 7. Detect & Broadcast ---
    chi2_thresh = chi2.ppf(1 - confidence, df=len(valid_cols))

    # Identify bad quarters
    bad_indices = np.where(distances > chi2_thresh)[0]
    bad_quarters = set()

    if len(bad_indices) > 0:
        q_ids = target_z_df["_quarter_id"].to_list()
        dates = target_z_df[date_col].to_list()

        for idx in bad_indices:
            q = q_ids[idx]
            bad_quarters.add(q)
            if len(logs) < MAX_CORRECTIONS_LOG:
                logs.append({
                    "ticker": ticker,
                    "date": dates[idx],  # Representative date
                    "quarter": q,
                    "error_type": "mahalanobis_outlier",
                    "dist": float(distances[idx]),
                    "threshold": chi2_thresh
                })

    # --- 8. Imputation (Broadcast back to Daily) ---
    if not bad_quarters:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Strategy:
    # 1. Take the Clean Target Quarterly Data
    # 2. Nullify values in Bad Quarters
    # 3. Forward Fill (fill bad quarter with previous quarter's value)
    # 4. Join back to Daily data on Quarter_ID

    # Create 'Corrected' Quarterly Map
    corrected_map = (
        target_z_df
        .sort(date_col)
        .select(["_quarter_id"] + valid_cols)
        .with_columns([
            pl.when(pl.col("_quarter_id").is_in(bad_quarters))
            .then(None)
            .otherwise(pl.col(c))
            .alias(c)
            for c in valid_cols
        ])
        .with_columns([
            pl.col(c).forward_fill().alias(f"{c}_corrected")
            for c in valid_cols
        ])
    )

    # Prepare Daily Data for Join
    # Add quarter_id to daily data
    daily_w_quarter = (
        working_lf
        .with_columns(
            (pl.col(date_col).dt.year().cast(pl.Utf8) + "-" +
             pl.col(date_col).dt.quarter().cast(pl.Utf8)).alias("_quarter_id")
        )
    )

    # Join and Overwrite
    # We join the corrected quarterly values onto the daily rows
    final_df = (
        daily_w_quarter
        .join(corrected_map, on="_quarter_id", how="left")
        .with_columns([
            # If we have a corrected value, use it. Else keep original.
            pl.coalesce([pl.col(f"{c}_corrected"), pl.col(c)]).alias(c)
            for c in valid_cols
        ])
        .select(working_lf.collect_schema().names())  # Restore original schema
    )

    return (final_df.collect() if not is_lazy else final_df, logs)