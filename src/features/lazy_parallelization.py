from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import gc

import polars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MEMORY FIX: Limit audit log entries per ticker for fair distribution
MAX_LOGS_PER_TICKER = 64

def consolidate_audit_logs(raw_logs: list) -> dict | list:
    """
    Consolidate audit logs from parallel processing.

    When processing multiple tickers in parallel, each ticker returns audit logs
    (often empty [] or {}). This function flattens and merges them appropriately.

    Args:
        raw_logs: List of audit logs from parallel_process_tickers

    Returns:
        Consolidated audit logs - either a single list or dict depending on structure

    Examples:
        [[], [], []] → []
        [[], [{"error": "..."}], []] → [{"error": "..."}]
        [{}, {"col1": [...]}, {}] → {"col1": [...]}
        [{"hard": [], "soft": []}, {"hard": [...], "soft": []}] → {"hard": [...], "soft": [...]}
    """
    if not raw_logs:
        return []

    # Check the structure of non-empty logs to determine consolidation strategy
    sample_non_empty = None
    for log in raw_logs:
        if log:  # Find first non-empty log
            sample_non_empty = log
            break

    # If all logs are empty, return appropriate empty structure
    if sample_non_empty is None:
        # Check if we have lists or dicts
        has_dict = any(isinstance(log, dict) for log in raw_logs)
        return {} if has_dict else []

    # Strategy 1: List of lists → flatten to single list
    if isinstance(sample_non_empty, list):
        consolidated = []
        for log in raw_logs:
            if isinstance(log, list) and log:
                consolidated.extend(log)
        return consolidated

    # Strategy 2: List of dicts → merge dicts
    if isinstance(sample_non_empty, dict):
        # Check if it's a hard/soft filter structure (financial_unequivalencies)
        has_hard_soft = "hard_filter_errors" in sample_non_empty or "soft_filter_warnings" in sample_non_empty

        if has_hard_soft:
            # Merge hard/soft filter logs
            consolidated = {"hard_filter_errors": [], "soft_filter_warnings": []}
            for log in raw_logs:
                if isinstance(log, dict):
                    if "hard_filter_errors" in log and log["hard_filter_errors"]:
                        consolidated["hard_filter_errors"].extend(log["hard_filter_errors"])
                    if "soft_filter_warnings" in log and log["soft_filter_warnings"]:
                        consolidated["soft_filter_warnings"].extend(log["soft_filter_warnings"])
            return consolidated
        else:
            # Merge dict with nested lists (negative_fundamentals structure)
            consolidated = {}
            for log in raw_logs:
                if isinstance(log, dict):
                    for key, value in log.items():
                        if key not in consolidated:
                            consolidated[key] = []
                        if isinstance(value, list) and value:
                            consolidated[key].extend(value)
            return consolidated

    # Fallback: return as-is
    return raw_logs

def truncate_ticker_logs(audit_log, max_entries: int = MAX_LOGS_PER_TICKER):
    """
    Truncate audit logs for a single ticker to prevent memory explosion.

    This ensures fair log distribution across all tickers - each ticker
    contributes at most max_entries to the final logs, rather than having
    early-processed tickers dominate the log output.

    Args:
        audit_log: Audit log from a single ticker (list, dict, or other)
        max_entries: Maximum number of log entries to keep per ticker

    Returns:
        Truncated audit log in the same structure as input
    """
    if isinstance(audit_log, list):
        # Simple list of log entries - truncate directly
        return audit_log[:max_entries]

    elif isinstance(audit_log, dict):
        # Check if it's a hard/soft filter structure (financial_unequivalencies)
        has_hard_soft = "hard_filter_errors" in audit_log or "soft_filter_warnings" in audit_log

        if has_hard_soft:
            # Truncate hard and soft filters separately, splitting the limit
            hard_limit = max_entries // 2
            soft_limit = max_entries - hard_limit

            return {
                "hard_filter_errors": audit_log.get("hard_filter_errors", [])[:hard_limit],
                "soft_filter_warnings": audit_log.get("soft_filter_warnings", [])[:soft_limit]
            }
        else:
            # Dict with nested lists (negative_fundamentals structure)
            # Distribute max_entries across all columns proportionally
            truncated = {}
            total_entries = sum(len(v) if isinstance(v, list) else 1 for v in audit_log.values())

            if total_entries == 0:
                return audit_log

            for key, value in audit_log.items():
                if isinstance(value, list):
                    # Proportional allocation based on original size
                    proportion = len(value) / total_entries
                    key_limit = max(1, int(max_entries * proportion))
                    truncated[key] = value[:key_limit]
                else:
                    truncated[key] = value

            return truncated

    # Fallback: return as-is for other types
    return audit_log


def process_single_ticker(lf: polars.LazyFrame,
                          columns: list[str],
                          ticker: str,
                          function: Callable,
                          metadata: polars.LazyFrame = None,
                          shared_data: dict = None) -> tuple[polars.LazyFrame, list[dict]]:
    """
    Process a single ticker file through any cleaning function.

    Args:
        lf: LazyFrame containing ticker data
        columns: List of column names to process
        ticker: Ticker symbol
        function: Validation/cleaning function to apply
        metadata: Optional LazyFrame containing metadata (e.g., sector information)
        shared_data: Optional dict containing data from all tickers (for cross-sectional analysis)

    Returns:
        Tuple of (ticker, cleaned_lazyframe, audit_log, metadata)
    """

    # Apply cleaning functions in sequence
    # Pass metadata and shared_data if the function signature supports it
    lf, audit = function(df=lf, columns=columns, ticker=ticker, metadata=metadata, shared_data=shared_data)

    return ticker, lf, audit, metadata


def parallel_process_tickers(
        data_dict: dict[str, tuple[polars.LazyFrame, polars.LazyFrame]],
        columns: list[str] = [""],
        function: Callable = None,
        max_workers: int = 8,
        batch_size: int = None,
        show_progress: bool = True,
        max_logs_per_ticker: int = MAX_LOGS_PER_TICKER,
        shared_data: dict = None
) -> tuple[dict[str, tuple[polars.LazyFrame, polars.LazyFrame]], list[dict]]:
    """
    Parallelize the processing of multiple tickers using threading.
    Threading works well with Polars since it releases the GIL.

    MEMORY OPTIMIZATIONS:
    - Per-ticker log truncation for fair distribution
    - Explicit garbage collection between batches
    - LazyFrames are kept lazy (not collected)

    Args:
        data_dict: Dictionary mapping ticker symbols to LazyFrames (or Tuples of LF, Metadata)
        columns: List of column names to process
        function: Validation/cleaning function to apply
        max_workers: Number of parallel threads
        batch_size: Number of tickers per batch
        show_progress: Whether to show progress bar
        max_logs_per_ticker: Maximum log entries per ticker (default: MAX_LOGS_PER_TICKER)
        shared_data: Optional dict containing data from all tickers (for cross-sectional analysis)

    Returns:
        Tuple of (cleaned_lazyframes_dict, all_audit_logs_list)
        - cleaned_lazyframes_dict preserves (LazyFrame, metadata) tuples
    """
    total_tickers = len(data_dict)
    cleaned_lazyframes = {}
    all_audit_logs = []
    failed_tickers = []
    total_logs_collected = 0
    truncated_ticker_count = 0

    if batch_size is None:
        batch_size = total_tickers

    logger.info(f"Starting parallel processing of {total_tickers} tickers with {max_workers} threads")
    logger.info(f"Per-ticker log limit: {max_logs_per_ticker} entries")

    ticker_keys = list(data_dict.keys())

    for batch_start in range(0, total_tickers, batch_size):
        batch_end = min(batch_start + batch_size, total_tickers)
        batch_tickers = ticker_keys[batch_start:batch_end]

        logger.info(f"Processing batch {batch_start // batch_size + 1}: "
                    f"tickers {batch_start} to {batch_end - 1} ({len(batch_tickers)} tickers)"
                    f"function: {str(function)}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}

            # --- TUPLE HANDLING: Extract (Data, Metadata) ---
            for ticker in batch_tickers:
                payload = data_dict[ticker]

                # Check if payload is a Tuple (Data, Metadata) and extract both
                if isinstance(payload, tuple):
                    lf_input = payload[0]
                    metadata = payload[1] if len(payload) > 1 else None
                else:
                    lf_input = payload
                    metadata = None

                future = executor.submit(
                    process_single_ticker,
                    lf_input,
                    columns,
                    ticker,
                    function,
                    metadata,
                    shared_data
                )
                future_to_ticker[future] = ticker
            # --- END TUPLE HANDLING ---

            futures = list(future_to_ticker.keys())
            iterator = tqdm(as_completed(futures), total=len(futures),
                            disable=not show_progress,
                            desc=f"Batch {batch_start // batch_size + 1}")

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    returned_ticker, cleaned_lf, audit_log, returned_metadata = future.result()
                    # Preserve metadata tuple for downstream filters
                    cleaned_lazyframes[returned_ticker] = (cleaned_lf, returned_metadata)

                    # MEMORY FIX: Truncate logs per ticker for fair distribution
                    original_size = get_log_size(audit_log)
                    truncated_log = truncate_ticker_logs(audit_log, max_logs_per_ticker)
                    truncated_size = get_log_size(truncated_log)

                    if truncated_size < original_size:
                        truncated_ticker_count += 1

                    all_audit_logs.append(truncated_log)
                    total_logs_collected += truncated_size

                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
                    failed_tickers.append(ticker)

        # MEMORY FIX: Force garbage collection after each batch
        gc.collect()

        logger.info(f"Batch complete. Total processed: {len(cleaned_lazyframes)}/{total_tickers}, "
                    f"Logs collected: {total_logs_collected}")

    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers: {failed_tickers}")

    if truncated_ticker_count > 0:
        logger.warning(f"Logs truncated for {truncated_ticker_count} tickers "
                       f"(limit: {max_logs_per_ticker} entries per ticker)")

    logger.info(f"Total audit logs collected: {total_logs_collected} entries across {len(cleaned_lazyframes)} tickers")

    return cleaned_lazyframes, all_audit_logs

def get_log_size(audit_log) -> int:
    """
    Calculate the number of log entries in an audit log structure.

    Args:
        audit_log: Audit log (list, dict, or other)

    Returns:
        Estimated number of log entries
    """
    if isinstance(audit_log, list):
        return len(audit_log)
    elif isinstance(audit_log, dict):
        return sum(len(v) if isinstance(v, list) else 1 for v in audit_log.values())
    else:
        return 1
