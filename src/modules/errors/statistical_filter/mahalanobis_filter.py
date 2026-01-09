import polars as pl
import numpy as np
import scipy.linalg
import threading
import warnings
from typing import Union, Optional
from sklearn.covariance import MinCovDet, LedoitWolf
from scipy.stats import chi2


class SectorModelCache:
    """
    Thread-safe cache for sector-level Mahalanobis models.

    This cache stores pre-computed MCD models for each sector, avoiding
    redundant computation when multiple tickers from the same sector
    are processed in parallel.
    """

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._computing = {}  # Track sectors currently being computed

    def get(self, sector: str) -> Optional[dict]:
        """Get cached model for a sector (thread-safe)."""
        with self._lock:
            return self._cache.get(sector)

    def set(self, sector: str, model: dict) -> None:
        """Store model for a sector (thread-safe)."""
        with self._lock:
            self._cache[sector] = model
            # Clear computing flag
            if sector in self._computing:
                del self._computing[sector]

    def get_or_compute(self, sector: str, compute_fn) -> Optional[dict]:
        """
        Get cached model or compute it if not present.

        Uses double-checked locking to ensure only one thread computes
        the model while others wait.
        """
        # Fast path: check if already cached
        model = self.get(sector)
        if model is not None:
            return model

        # Slow path: need to potentially compute
        with self._lock:
            # Double-check after acquiring lock
            if sector in self._cache:
                return self._cache[sector]

            # Check if another thread is computing
            if sector in self._computing:
                # Wait for the other thread by releasing and re-acquiring
                event = self._computing[sector]
            else:
                # We'll compute it - create event for others to wait on
                event = threading.Event()
                self._computing[sector] = event
                event = None  # Signal that we should compute

        if event is not None:
            # Another thread is computing - wait for it
            event.wait(timeout=300)  # 5 minute timeout
            return self.get(sector)

        # We're the computing thread
        try:
            model = compute_fn()
            if model is not None:
                self.set(sector, model)
            return model
        finally:
            # Signal completion to waiting threads
            with self._lock:
                if sector in self._computing:
                    self._computing[sector].set()
                    del self._computing[sector]

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        """Return number of cached sectors."""
        with self._lock:
            return len(self._cache)


def _build_sector_quarterly_data(
        peer_symbols: list[str],
        available_cols: list[str],
        date_col: str,
        shared_data: dict
) -> pl.DataFrame:
    """
    Build quarterly aggregated data for all peers in a sector.

    This collects data from shared_data for all peer symbols and
    collapses daily data to quarterly snapshots.
    """
    quarterly_frames = []

    for filename, payload in shared_data.items():
        # Skip the cache entry itself
        if filename == "__sector_model_cache__":
            continue

        symbol = filename.replace(".csv", "")
        if symbol in peer_symbols:
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

    if len(quarterly_frames) < 2:
        return None

    return pl.concat(quarterly_frames, how="diagonal")


def _normalize_quarterly_data(
        pooled_q_df: pl.DataFrame,
        available_cols: list[str]
) -> tuple[pl.DataFrame, list[str], list[str]]:
    """
    Apply time-aware normalization (Z-scores per quarter).

    Returns normalized dataframe, valid columns, and z-score column names.
    """
    # Filter only cols that exist in pooled data
    valid_cols = [c for c in available_cols if c in pooled_q_df.columns]

    z_exprs = []
    MAD_FACTOR = 1.4826

    for col in valid_cols:
        median_expr = pl.col(col).median().over("_quarter_id")
        mad_expr = (pl.col(col) - median_expr).abs().median().over("_quarter_id") * MAD_FACTOR

        safe_scale = (
            pl.when(mad_expr < 1e-6)
            .then(pl.col(col).abs().mean().over("_quarter_id") * 0.01 + 1e-6)
            .otherwise(mad_expr)
        )

        z_exprs.append(
            ((pl.col(col) - median_expr) / safe_scale).alias(f"_z_{col}")
        )

    # Apply normalization and filter quarters with >= 3 tickers
    normalized_df = (
        pooled_q_df
        .with_columns(pl.col("ticker").count().over("_quarter_id").alias("_peer_count"))
        .filter(pl.col("_peer_count") >= 3)
        .with_columns(z_exprs)
    )

    z_cols = [f"_z_{c}" for c in valid_cols]
    return normalized_df, valid_cols, z_cols


def _train_mcd_model(
        normalized_df: pl.DataFrame,
        z_cols: list[str],
        valid_cols: list[str]
) -> Optional[dict]:
    """
    Train robust covariance model on normalized sector data.

    Uses MCD (Minimum Covariance Determinant) as primary estimator with
    LedoitWolf shrinkage as fallback for numerical stability issues.

    Returns dict with robust_loc, robust_prec, or None if training fails.
    """
    training_matrix = normalized_df.select(z_cols).drop_nulls().to_numpy()

    # Safety: Need enough total points
    if len(training_matrix) < len(valid_cols) * 5:
        return None

    robust_cov = None
    robust_loc = None

    # Try MCD first, catch determinant warnings and fall back to LedoitWolf
    try:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")

            mcd = MinCovDet(random_state=42, support_fraction=0.9, assume_centered=False)
            mcd.fit(training_matrix)

            # Check if determinant warning was raised
            determinant_warning = any(
                "Determinant has increased" in str(w.message)
                for w in caught_warnings
            )

            if not determinant_warning:
                robust_cov = mcd.covariance_
                robust_loc = mcd.location_

    except Exception:
        pass  # Will fall through to LedoitWolf

    # Fallback to LedoitWolf if MCD failed or had numerical issues
    if robust_cov is None:
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(training_matrix)

            robust_cov = lw.covariance_
            robust_loc = lw.location_
        except Exception:
            return None

    # Regularize diagonal for numerical stability (invertibility)
    robust_cov.flat[::robust_cov.shape[0] + 1] += 1e-6
    robust_prec = scipy.linalg.pinvh(robust_cov)

    return {
        "robust_loc": robust_loc,
        "robust_prec": robust_prec
    }


def _compute_sector_model(
        sector: str,
        peer_symbols: list[str],
        available_cols: list[str],
        date_col: str,
        confidence: float,
        shared_data: dict
) -> Optional[dict]:
    """
    Compute the complete sector model (quarterly data + MCD).

    This is the expensive computation that gets cached per sector.
    """
    # Build quarterly data for all sector peers
    pooled_q_df = _build_sector_quarterly_data(
        peer_symbols, available_cols, date_col, shared_data
    )

    if pooled_q_df is None:
        return None

    # Normalize the data
    normalized_df, valid_cols, z_cols = _normalize_quarterly_data(
        pooled_q_df, available_cols
    )

    if normalized_df.is_empty():
        return None

    # Train MCD model
    mcd_result = _train_mcd_model(normalized_df, z_cols, valid_cols)

    if mcd_result is None:
        return None

    # Compute chi-squared threshold
    chi2_thresh = chi2.ppf(1 - confidence, df=len(valid_cols))

    return {
        "normalized_df": normalized_df,
        "valid_cols": valid_cols,
        "z_cols": z_cols,
        "robust_loc": mcd_result["robust_loc"],
        "robust_prec": mcd_result["robust_prec"],
        "chi2_thresh": chi2_thresh,
        "date_col": date_col
    }


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
    Detect outliers using 'Sector Cloud' approach with cached sector models.

    This optimized version caches the MCD model per sector, so tickers in
    the same sector share the computed model instead of recomputing it.

    Algorithm:
    1. Collapses Daily -> Quarterly (Fixes Singularity)
    2. Normalizes per Quarter (Fixes Time Alignment)
    3. Pools all history to learn Sector Geometry (CACHED per sector)
    4. Scores target ticker against sector model
    5. Imputes outlier quarters with forward-filled values
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

    # --- 3. Get or Compute Cached Sector Model ---
    cache = shared_data.get("__sector_model_cache__")

    if cache is not None:
        # Use cached sector model (optimized path)
        def compute_fn():
            return _compute_sector_model(
                sector=target_sector,
                peer_symbols=peer_symbols,
                available_cols=available_cols,
                date_col=date_col,
                confidence=confidence,
                shared_data=shared_data
            )

        sector_model = cache.get_or_compute(target_sector, compute_fn)
    else:
        # No cache available - compute directly (fallback for backward compatibility)
        sector_model = _compute_sector_model(
            sector=target_sector,
            peer_symbols=peer_symbols,
            available_cols=available_cols,
            date_col=date_col,
            confidence=confidence,
            shared_data=shared_data
        )

    if sector_model is None:
        logs.append({"ticker": ticker, "error_type": "sector_model_fail", "message": "Could not build sector model"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # --- 4. Extract Cached Model Components ---
    normalized_df = sector_model["normalized_df"]
    valid_cols = sector_model["valid_cols"]
    z_cols = sector_model["z_cols"]
    robust_loc = sector_model["robust_loc"]
    robust_prec = sector_model["robust_prec"]
    chi2_thresh = sector_model["chi2_thresh"]

    # --- 5. Score Target Quarters ---
    target_z_df = normalized_df.filter(pl.col("ticker") == ticker_symbol)

    if target_z_df.is_empty():
        logs.append({"ticker": ticker, "error_type": "insufficient_history", "message": "Not enough overlapping peers"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    target_mat = target_z_df.select(z_cols).to_numpy()

    # Calculate Distances
    distances = np.full(len(target_z_df), np.nan)
    mask = np.isfinite(target_mat).all(axis=1)

    if np.any(mask):
        diff = target_mat[mask] - robust_loc
        # (x-u)P(x-u)'
        distances[mask] = np.sum((diff @ robust_prec) * diff, axis=1)

    # --- 6. Detect Outliers ---
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
                    "date": dates[idx],
                    "quarter": q,
                    "error_type": "mahalanobis_outlier",
                    "dist": float(distances[idx]),
                    "threshold": chi2_thresh
                })

    # --- 7. Imputation (Broadcast back to Daily) ---
    if not bad_quarters:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

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
    daily_w_quarter = (
        working_lf
        .with_columns(
            (pl.col(date_col).dt.year().cast(pl.Utf8) + "-" +
             pl.col(date_col).dt.quarter().cast(pl.Utf8)).alias("_quarter_id")
        )
    )

    # Join and Overwrite
    final_df = (
        daily_w_quarter
        .join(corrected_map, on="_quarter_id", how="left")
        .with_columns([
            pl.coalesce([pl.col(f"{c}_corrected"), pl.col(c)]).alias(c)
            for c in valid_cols
        ])
        .select(working_lf.collect_schema().names())
    )

    return (final_df.collect() if not is_lazy else final_df, logs)