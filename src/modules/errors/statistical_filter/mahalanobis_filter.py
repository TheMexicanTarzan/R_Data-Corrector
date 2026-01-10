import polars
import numpy
import scipy.linalg
import warnings
from typing import Union, Optional
from sklearn.covariance import MinCovDet, LedoitWolf
from scipy.stats import chi2

MAX_CORRECTIONS_LOG = 5

class MetadataCache:
    """
    OPTIMIZATION: Pre-computed metadata lookups.

    Instead of filtering metadata LazyFrame for each ticker (O(n) per lookup),
    we build O(1) lookup dictionaries once and reuse them across all tickers.

    Simplified version without thread locks since we process sectors serially.
    """

    def __init__(self):
        self._symbol_to_sector: dict[str, str] = {}
        self._sector_to_symbols: dict[str, list[str]] = {}
        self._initialized = False

    def initialize(self, metadata: polars.LazyFrame) -> None:
        """Build lookup dictionaries from metadata LazyFrame (called once)."""
        if self._initialized:
            return

        try:
            meta_df = metadata.select(["symbol", "sector"]).collect()

            for row in meta_df.iter_rows():
                symbol, sector = row[0], row[1]
                if symbol is None or sector is None:
                    continue

                # Clean symbol (remove .csv suffix if present)
                clean_symbol = symbol.replace(".csv", "") if symbol.endswith(".csv") else symbol

                self._symbol_to_sector[clean_symbol] = sector

                if sector not in self._sector_to_symbols:
                    self._sector_to_symbols[sector] = []
                self._sector_to_symbols[sector].append(clean_symbol)

            self._initialized = True
        except Exception:
            pass

    def get_sector(self, symbol: str) -> Optional[str]:
        """O(1) lookup of sector for a symbol."""
        clean_symbol = symbol.replace(".csv", "") if symbol.endswith(".csv") else symbol
        return self._symbol_to_sector.get(clean_symbol)

    def get_peer_symbols(self, sector: str) -> list[str]:
        """O(1) lookup of all symbols in a sector."""
        return self._sector_to_symbols.get(sector, [])

    def is_initialized(self) -> bool:
        return self._initialized


# OPTIMIZATION: Pre-compute quarter ID as integer (faster than string)
def _compute_quarter_id_int(year: int, quarter: int) -> int:
    """Compute quarter ID as integer: YYYY * 10 + Q (e.g., 2024Q1 = 20241)."""
    return year * 10 + quarter


def _build_sector_quarterly_data(
        peer_symbols: list[str],
        available_cols: list[str],
        date_col: str,
        shared_data: dict,
        schema_cache: dict = None
) -> Optional[polars.DataFrame]:
    """
    OPTIMIZED: Builds quarterly data using Lazy execution and Direct Lookup.

    Key optimizations:
    1. Direct dict lookup O(K) instead of scan O(N)
    2. Schema caching to avoid repeated collect_schema() calls
    3. Integer quarter IDs (faster than string concatenation)
    4. Single materialization at the end
    """
    lazy_frames = []

    if schema_cache is None:
        schema_cache = {}

    for symbol in peer_symbols:
        payload = shared_data.get(symbol) or shared_data.get(f"{symbol}.csv")

        if payload is None:
            continue

        peer_lf = payload[0] if isinstance(payload, tuple) else payload

        try:
            # OPTIMIZATION: Cache schema lookups
            lf_id = id(peer_lf)
            if lf_id in schema_cache:
                peer_schema = schema_cache[lf_id]
            else:
                peer_schema = set(peer_lf.collect_schema().names())
                schema_cache[lf_id] = peer_schema

            cols_curr = [c for c in [date_col] + available_cols if c in peer_schema]

            if len(cols_curr) > 1:
                # OPTIMIZATION: Use integer quarter ID instead of string
                q_peer = (
                    peer_lf
                    .select(cols_curr)
                    .with_columns([
                        polars.lit(symbol).alias("ticker"),
                        (polars.col(date_col).dt.year() * 10 +
                         polars.col(date_col).dt.quarter()).alias("_quarter_id")
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

    return polars.concat(lazy_frames, how="diagonal").collect()


def _normalize_quarterly_data(
        pooled_q_df: polars.DataFrame,
        available_cols: list[str]
) -> tuple[polars.DataFrame, list[str], list[str]]:
    """
    OPTIMIZED: Normalize data using MAD-based z-scores.

    Builds all expressions at once for better Polars optimization.
    """
    valid_cols = [c for c in available_cols if c in pooled_q_df.columns]
    MAD_FACTOR = 1.4826

    # OPTIMIZATION: Build all expressions in one list comprehension
    z_exprs = []
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
    """
    OPTIMIZED: Train MCD model with reduced memory allocations.
    """
    training_matrix = normalized_df.select(z_cols).drop_nulls().to_numpy()
    n_samples, n_features = training_matrix.shape

    if n_samples < n_features * 5:
        return None

    robust_cov = None
    robust_loc = None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=".*Determinant has increased.*")

            rng = numpy.random.RandomState(42)

            # OPTIMIZATION: Compute stds and add noise in-place
            col_stds = numpy.std(training_matrix, axis=0)
            col_stds[col_stds == 0] = 1.0

            # OPTIMIZATION: Pre-allocate and use in-place operations
            noise = rng.normal(0, 1e-5, training_matrix.shape)
            noise *= col_stds  # In-place multiply
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

    # Add regularization to diagonal (in-place)
    robust_cov.flat[::robust_cov.shape[0] + 1] += 1e-6

    try:
        robust_prec = scipy.linalg.pinvh(robust_cov)
    except Exception:
        return None

    return {
        "robust_loc": robust_loc,
        "robust_prec": robust_prec
    }


def compute_sector_model(
        sector: str,
        peer_symbols: list[str],
        available_cols: list[str],
        date_col: str,
        confidence: float,
        shared_data: dict,
        schema_cache: dict = None
) -> Optional[dict]:
    """
    Compute sector model for Mahalanobis filtering.

    Simplified version without caching - called once per sector in serial mode.
    Pre-computes ticker z-score data for O(1) ticker lookups during filtering.
    """
    pooled_q_df = _build_sector_quarterly_data(
        peer_symbols, available_cols, date_col, shared_data, schema_cache
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

    # Pre-compute per-ticker z-score data for O(1) lookup during filtering
    ticker_z_data = {}
    for ticker in normalized_df["ticker"].unique().to_list():
        ticker_df = normalized_df.filter(polars.col("ticker") == ticker)
        if not ticker_df.is_empty():
            ticker_z_data[ticker] = {
                "z_matrix": ticker_df.select(z_cols).to_numpy(),
                "quarter_ids": ticker_df["_quarter_id"].to_list(),
                "dates": ticker_df[date_col].to_list() if date_col in ticker_df.columns else [],
                "df": ticker_df
            }

    return {
        "normalized_df": normalized_df,
        "valid_cols": valid_cols,
        "z_cols": z_cols,
        "robust_loc": mcd_result["robust_loc"],
        "robust_prec": mcd_result["robust_prec"],
        "chi2_thresh": chi2_thresh,
        "date_col": date_col,
        "ticker_z_data": ticker_z_data,
        "sector": sector,
        "peer_symbols": peer_symbols
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
    """
    Mahalanobis filter using pre-computed sector model.

    Simplified version for sector-based serial processing:
    - Expects pre-computed sector model in shared_data["__sector_model__"]
    - No caching needed since each sector is processed once
    - Parallelizes filtering within the sector's tickers
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()
    logs = []

    ticker_symbol = ticker.replace(".csv", "") if ticker.endswith(".csv") else ticker

    # Get schema
    schema_cache = shared_data.get("__schema_cache__") if shared_data else None
    if schema_cache is None and shared_data is not None:
        schema_cache = {}
        shared_data["__schema_cache__"] = schema_cache

    lf_id = id(working_lf)
    if schema_cache is not None and lf_id in schema_cache:
        schema_cols = schema_cache[lf_id]
    else:
        schema_cols = set(working_lf.collect_schema().names())
        if schema_cache is not None:
            schema_cache[lf_id] = schema_cols

    if date_col not in schema_cols:
        logs.append({"ticker": ticker, "error_type": "missing_date", "message": "Date col missing"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    available_cols = [col for col in columns if col in schema_cols]
    if len(available_cols) < 2:
        logs.append({"ticker": ticker, "error_type": "insufficient_cols", "message": "<2 cols"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    if shared_data is None:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Get pre-computed sector model from shared_data
    sector_model = shared_data.get("__sector_model__")
    if sector_model is None:
        logs.append({"ticker": ticker, "error_type": "no_sector_model", "message": "No pre-computed sector model"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    valid_cols = sector_model["valid_cols"]
    z_cols = sector_model["z_cols"]
    robust_loc = sector_model["robust_loc"]
    robust_prec = sector_model["robust_prec"]
    chi2_thresh = sector_model["chi2_thresh"]
    ticker_z_data = sector_model.get("ticker_z_data", {})

    # OPTIMIZATION: Use pre-computed ticker data if available
    if ticker_symbol in ticker_z_data:
        ticker_data = ticker_z_data[ticker_symbol]
        target_mat = ticker_data["z_matrix"]
        q_ids = ticker_data["quarter_ids"]
        dates = ticker_data["dates"]
        target_z_df = ticker_data["df"]
    else:
        # Fallback to original method
        normalized_df = sector_model["normalized_df"]
        target_z_df = normalized_df.filter(polars.col("ticker") == ticker_symbol)

        if target_z_df.is_empty():
            logs.append({"ticker": ticker, "error_type": "insufficient_history", "message": "Not enough overlapping peers"})
            return (working_lf.collect() if not is_lazy else working_lf, logs)

        target_mat = target_z_df.select(z_cols).to_numpy()
        q_ids = target_z_df["_quarter_id"].to_list()
        dates = target_z_df[date_col].to_list() if date_col in target_z_df.columns else []

    if target_mat is None or len(target_mat) == 0:
        logs.append({"ticker": ticker, "error_type": "insufficient_history", "message": "Not enough overlapping peers"})
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # OPTIMIZATION: Vectorized distance computation with pre-allocated array
    n_rows = len(target_mat)
    distances = numpy.full(n_rows, numpy.nan, dtype=numpy.float64)
    mask = numpy.isfinite(target_mat).all(axis=1)

    if numpy.any(mask):
        diff = target_mat[mask] - robust_loc
        # Efficient Mahalanobis: sum((diff @ precision) * diff, axis=1)
        distances[mask] = numpy.einsum('ij,jk,ik->i', diff, robust_prec, diff)

    bad_indices = numpy.where(distances > chi2_thresh)[0]
    bad_quarters = set()

    # Collect ALL outliers first, then sort by severity (distance) to return top K
    all_outlier_logs = []

    if len(bad_indices) > 0:
        for idx in bad_indices:
            q = q_ids[idx]
            bad_quarters.add(q)
            if dates:
                all_outlier_logs.append({
                    "ticker": ticker,
                    "date": dates[idx] if idx < len(dates) else None,
                    "quarter": q,
                    "error_type": "mahalanobis_outlier",
                    "dist": float(distances[idx]),
                    "threshold": chi2_thresh,
                    "_severity": float(distances[idx])  # For sorting
                })

        # Sort by severity (highest distance first) and keep top K
        all_outlier_logs.sort(key=lambda x: x["_severity"], reverse=True)
        for log_entry in all_outlier_logs[:MAX_CORRECTIONS_LOG]:
            del log_entry["_severity"]  # Remove internal field
            logs.append(log_entry)

    if not bad_quarters:
        return (working_lf.collect() if not is_lazy else working_lf, logs)

    # Build correction map using target_z_df
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

    # OPTIMIZATION: Use integer quarter ID (matches what we stored)
    daily_w_quarter = (
        working_lf
        .with_columns(
            (polars.col(date_col).dt.year() * 10 +
             polars.col(date_col).dt.quarter()).alias("_quarter_id")
        )
    )

    # Get original columns list efficiently
    orig_cols = list(schema_cols)

    final_df = (
        daily_w_quarter
        .join(corrected_map, on="_quarter_id", how="left")
        .with_columns([
            polars.coalesce([polars.col(f"{c}_corrected"), polars.col(c)]).alias(c)
            for c in valid_cols
        ])
        .select(orig_cols)
    )

    return (final_df.collect() if not is_lazy else final_df, logs)
