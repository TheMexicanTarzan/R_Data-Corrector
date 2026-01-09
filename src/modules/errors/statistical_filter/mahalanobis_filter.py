import polars
import numpy
import scipy.linalg
import threading
import warnings
from typing import Union, Optional
from sklearn.covariance import MinCovDet, LedoitWolf
from scipy.stats import chi2


class SectorModelCache:
    """Thread-safe cache for sector-level Mahalanobis models."""

    def __init__(self):
        self._cache = {}
        self._lock = threading.Lock()
        self._computing = {}

    def get(self, sector: str) -> Optional[dict]:
        with self._lock:
            return self._cache.get(sector)

    def set(self, sector: str, model: dict) -> None:
        with self._lock:
            self._cache[sector] = model
            if sector in self._computing:
                del self._computing[sector]

    def get_or_compute(self, sector: str, compute_fn) -> Optional[dict]:
        model = self.get(sector)
        if model is not None:
            return model

        with self._lock:
            if sector in self._cache:
                return self._cache[sector]
            if sector in self._computing:
                event = self._computing[sector]
            else:
                event = threading.Event()
                self._computing[sector] = event
                event = None

        if event is not None:
            event.wait(timeout=300)
            return self.get(sector)

        try:
            model = compute_fn()
            if model is not None:
                self.set(sector, model)
            return model
        finally:
            with self._lock:
                if sector in self._computing:
                    self._computing[sector].set()
                    del self._computing[sector]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)


def _build_sector_quarterly_data(
        peer_symbols: list[str],
        available_cols: list[str],
        date_col: str,
        shared_data: dict
) -> Optional[polars.DataFrame]:
    """
    OPTIMIZED: Builds quarterly data using Lazy execution and Direct Lookup.
    """
    lazy_frames = []

    # 1. OPTIMIZATION: Direct Dict Lookup (O(K)) instead of Scan (O(N))
    # We iterate only the peers we need, not the entire database.
    # We try both "SYMBOL" and "SYMBOL.csv" as keys to be safe.
    for symbol in peer_symbols:
        payload = shared_data.get(symbol) or shared_data.get(f"{symbol}.csv")

        if payload is None:
            continue

        # Handle tuple payload if necessary
        peer_lf = payload[0] if isinstance(payload, tuple) else payload

        # 2. OPTIMIZATION: Keep it Lazy!
        # Do not call .collect() here. Just build the query plan.
        try:
            # Note: collect_schema() is fast, but if schema is known/fixed,
            # hardcoding columns avoids even this IO overhead.
            peer_schema = set(peer_lf.collect_schema().names())
            cols_curr = [c for c in [date_col] + available_cols if c in peer_schema]

            if len(cols_curr) > 1:
                q_peer = (
                    peer_lf
                    .select(cols_curr)
                    .with_columns([
                        polars.lit(symbol).alias("ticker"),
                        (polars.col(date_col).dt.year().cast(polars.Utf8) + "-" +
                         polars.col(date_col).dt.quarter().cast(polars.Utf8)).alias("_quarter_id")
                    ])
                    .group_by(["ticker", "_quarter_id"])
                    .agg([
                        polars.col(date_col).last(),
                        *[polars.col(c).last() for c in cols_curr if c != date_col]
                    ])
                )
                lazy_frames.append(q_peer)
        except Exception:
            continue

    if len(lazy_frames) < 2:
        return None

    # 3. OPTIMIZATION: Single Materialization
    # Concatenate all LazyFrames and run the query engine ONCE.
    # This allows Polars to optimize IO and threading across all files.
    return polars.concat(lazy_frames, how="diagonal").collect()


def _normalize_quarterly_data(
        pooled_q_df: polars.DataFrame,
        available_cols: list[str]
) -> tuple[polars.DataFrame, list[str], list[str]]:
    valid_cols = [c for c in available_cols if c in pooled_q_df.columns]
    z_exprs = []
    MAD_FACTOR = 1.4826

    for col in valid_cols:
        median_expr = polars.col(col).median().over("_quarter_id")
        mad_expr = (polars.col(col) - median_expr).abs().median().over("_quarter_id") * MAD_FACTOR

        safe_scale = (
            polars.when(mad_expr < 1e-6)
            .then(polars.col(col).abs().mean().over("_quarter_id") * 0.01 + 1e-6)
            .otherwise(mad_expr)
        )

        z_exprs.append(((polars.col(col) - median_expr) / safe_scale).alias(f"_z_{col}"))

    normalized_df = (
        pooled_q_df
        .with_columns(polars.col("ticker").count().over("_quarter_id").alias("_peer_count"))
        .filter(polars.col("_peer_count") >= 3)
        .with_columns(z_exprs)
    )

    z_cols = [f"_z_{c}" for c in valid_cols]
    return normalized_df, valid_cols, z_cols


def _train_mcd_model(
        normalized_df: polars.DataFrame,
        z_cols: list[str],
        valid_cols: list[str]
) -> Optional[dict]:
    training_matrix = normalized_df.select(z_cols).drop_nulls().to_numpy()

    if len(training_matrix) < len(valid_cols) * 5:
        return None

    robust_cov = None
    robust_loc = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*Determinant has increased.*")

            rng = numpy.random.RandomState(42)
            col_stds = numpy.std(training_matrix, axis=0)
            col_stds[col_stds == 0] = 1.0

            noise = rng.normal(0, 1e-5, training_matrix.shape) * col_stds
            training_matrix_stable = training_matrix + noise

            mcd = MinCovDet(random_state=42, support_fraction=0.9, assume_centered=False)
            mcd.fit(training_matrix_stable)

            robust_cov = mcd.covariance_
            robust_loc = mcd.location_

    except (RuntimeWarning, ValueError, Exception):
        robust_cov = None

    if robust_cov is None:
        try:
            lw = LedoitWolf(assume_centered=False)
            lw.fit(training_matrix)
            robust_cov = lw.covariance_
            robust_loc = lw.location_
        except Exception:
            return None

    robust_cov.flat[::robust_cov.shape[0] + 1] += 1e-6

    try:
        robust_prec = scipy.linalg.pinvh(robust_cov)
    except Exception:
        return None

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
    pooled_q_df = _build_sector_quarterly_data(
        peer_symbols, available_cols, date_col, shared_data
    )

    if pooled_q_df is None:
        return None

    normalized_df, valid_cols, z_cols = _normalize_quarterly_data(
        pooled_q_df, available_cols
    )

    if normalized_df.is_empty():
        return None

    mcd_result = _train_mcd_model(normalized_df, z_cols, valid_cols)

    if mcd_result is None:
        return None

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
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        confidence: float = 0.0001,
        shared_data: dict = None
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()
    logs = []
    MAX_CORRECTIONS_LOG = 50

    ticker_symbol = ticker.replace(".csv", "") if ticker.endswith(".csv") else ticker

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

    try:
        meta_df = metadata.filter(polars.col("symbol") == ticker_symbol).select("sector").collect()
        if meta_df.is_empty() or meta_df["sector"][0] is None:
            return (working_lf.collect() if not is_lazy else working_lf, logs)

        target_sector = meta_df["sector"][0]
        peer_symbols = metadata.filter(polars.col("sector") == target_sector).select("symbol").collect()[
            "symbol"].to_list()
    except Exception as e:
        logs.append({"ticker": ticker, "error_type": "meta_error", "message": str(e)})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    cache = shared_data.get("__sector_model_cache__")

    if cache is not None:
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

    normalized_df = sector_model["normalized_df"]
    valid_cols = sector_model["valid_cols"]
    z_cols = sector_model["z_cols"]
    robust_loc = sector_model["robust_loc"]
    robust_prec = sector_model["robust_prec"]
    chi2_thresh = sector_model["chi2_thresh"]

    target_z_df = normalized_df.filter(polars.col("ticker") == ticker_symbol)

    if target_z_df.is_empty():
        logs.append({"ticker": ticker, "error_type": "insufficient_history", "message": "Not enough overlapping peers"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    target_mat = target_z_df.select(z_cols).to_numpy()

    distances = numpy.full(len(target_z_df), numpy.nan)
    mask = numpy.isfinite(target_mat).all(axis=1)

    if numpy.any(mask):
        diff = target_mat[mask] - robust_loc
        distances[mask] = numpy.sum((diff @ robust_prec) * diff, axis=1)

    bad_indices = numpy.where(distances > chi2_thresh)[0]
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

    if not bad_quarters:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    corrected_map = (
        target_z_df
        .sort(date_col)
        .select(["_quarter_id"] + valid_cols)
        .with_columns([
            polars.when(polars.col("_quarter_id").is_in(bad_quarters))
            .then(None)
            .otherwise(polars.col(c))
            .alias(c)
            for c in valid_cols
        ])
        .with_columns([
            polars.col(c).forward_fill().alias(f"{c}_corrected")
            for c in valid_cols
        ])
    )

    daily_w_quarter = (
        working_lf
        .with_columns(
            (polars.col(date_col).dt.year().cast(polars.Utf8) + "-" +
             polars.col(date_col).dt.quarter().cast(polars.Utf8)).alias("_quarter_id")
        )
    )

    final_df = (
        daily_w_quarter
        .join(corrected_map, on="_quarter_id", how="left")
        .with_columns([
            polars.coalesce([polars.col(f"{c}_corrected"), polars.col(c)]).alias(c)
            for c in valid_cols
        ])
        .select(working_lf.collect_schema().names())
    )

    return (final_df.collect() if not is_lazy else final_df, logs)