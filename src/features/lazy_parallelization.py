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

# MEMORY FIX: Smaller batch size for reduced memory pressure
DEFAULT_BATCH_SIZE = 256


def truncate_ticker_logs(audit_log, max_entries: int = MAX_LOGS_PER_TICKER):
    """
    Truncate audit logs for a single ticker to prevent memory explosion.

    This ensures fair log distribution across all tickers - each ticker
    contributes at most max_entries to the final logs, rather than having
    early-processed tickers dominate the log output.
    """
    if isinstance(audit_log, list):
        return audit_log[:max_entries]

    elif isinstance(audit_log, dict):
        has_hard_soft = "hard_filter_errors" in audit_log or "soft_filter_warnings" in audit_log

        if has_hard_soft:
            hard_limit = max_entries // 2
            soft_limit = max_entries - hard_limit
            return {
                "hard_filter_errors": audit_log.get("hard_filter_errors", [])[:hard_limit],
                "soft_filter_warnings": audit_log.get("soft_filter_warnings", [])[:soft_limit]
            }
        else:
            truncated = {}
            total_entries = sum(len(v) if isinstance(v, list) else 1 for v in audit_log.values())

            if total_entries == 0:
                return audit_log

            for key, value in audit_log.items():
                if isinstance(value, list):
                    proportion = len(value) / total_entries
                    key_limit = max(1, int(max_entries * proportion))
                    truncated[key] = value[:key_limit]
                else:
                    truncated[key] = value

            return truncated

    return audit_log


def get_log_size(audit_log) -> int:
    """Calculate the number of log entries in an audit log structure."""
    if isinstance(audit_log, list):
        return len(audit_log)
    elif isinstance(audit_log, dict):
        return sum(len(v) if isinstance(v, list) else 1 for v in audit_log.values())
    else:
        return 1


def process_single_ticker(lf: polars.LazyFrame,
                          columns: list[str],
                          ticker: str,
                          function: Callable,
                          metadata: polars.LazyFrame = None,
                          shared_data: dict = None) -> tuple[polars.LazyFrame, list[dict]]:
    """
    Process a single ticker file through any cleaning function.
    Returns only the cleaned LazyFrame and audit log (no ticker name to reduce memory).
    """
    lf, audit = function(df=lf, columns=columns, ticker=ticker, metadata=metadata, shared_data=shared_data)
    return lf, audit, metadata


def parallel_process_tickers_inplace(
        data_dict: dict[str, tuple[polars.LazyFrame, polars.LazyFrame]],
        columns: list[str] = [""],
        function: Callable = None,
        max_workers: int = 8,
        batch_size: int = None,
        show_progress: bool = True,
        max_logs_per_ticker: int = MAX_LOGS_PER_TICKER,
        shared_data: dict = None
) -> list[dict]:
    """
    Parallelize the processing of multiple tickers using threading - IN-PLACE.

    MEMORY OPTIMIZATION: Instead of creating a new dictionary with all results,
    this function mutates data_dict in-place, replacing entries as they complete.
    This prevents having two copies of the dataset in memory.

    Returns:
        Only the audit logs (the data_dict is modified in-place)
    """
    total_tickers = len(data_dict)
    all_audit_logs = []
    failed_tickers = []

    if batch_size is None:
        batch_size = min(DEFAULT_BATCH_SIZE, total_tickers)

    logger.info(f"Processing {total_tickers} tickers IN-PLACE with {max_workers} threads (batch size: {batch_size})")

    ticker_keys = list(data_dict.keys())

    for batch_start in range(0, total_tickers, batch_size):
        batch_end = min(batch_start + batch_size, total_tickers)
        batch_tickers = ticker_keys[batch_start:batch_end]
        batch_num = batch_start // batch_size + 1
        total_batches = (total_tickers + batch_size - 1) // batch_size

        logger.info(f"Batch {batch_num}/{total_batches}: processing {len(batch_tickers)} tickers")

        # MEMORY FIX: Process and update in smaller sub-batches
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}

            for ticker in batch_tickers:
                payload = data_dict[ticker]

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

            # Process completed futures immediately and update dict
            iterator = tqdm(
                as_completed(future_to_ticker.keys()),
                total=len(future_to_ticker),
                disable=not show_progress,
                desc=f"Batch {batch_num}"
            )

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    cleaned_lf, audit_log, returned_metadata = future.result()

                    # MEMORY FIX: Update in-place immediately
                    data_dict[ticker] = (cleaned_lf, returned_metadata)

                    # MEMORY FIX: Truncate and store log immediately
                    truncated_log = truncate_ticker_logs(audit_log, max_logs_per_ticker)
                    if get_log_size(truncated_log) > 0:
                        all_audit_logs.append(truncated_log)

                    # MEMORY FIX: Clear future result reference
                    del cleaned_lf, audit_log, returned_metadata

                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
                    failed_tickers.append(ticker)

            # MEMORY FIX: Clear future references after batch
            future_to_ticker.clear()

        # MEMORY FIX: Force garbage collection after each batch
        gc.collect()

    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers")

    logger.info(f"Completed: {total_tickers - len(failed_tickers)}/{total_tickers} tickers processed")

    return all_audit_logs


def parallel_process_tickers_inplace_sector(
        data_dict: dict[str, tuple[polars.LazyFrame, polars.LazyFrame]],
        ticker_keys: list[str],
        columns: list[str] = [""],
        function: Callable = None,
        max_workers: int = 8,
        batch_size: int = None,
        show_progress: bool = True,
        max_logs_per_ticker: int = MAX_LOGS_PER_TICKER,
        shared_data: dict = None
) -> list[dict]:
    """
    Process a specific subset of tickers (for sector-based processing) IN-PLACE.

    This is used by Mahalanobis filtering which processes tickers by sector.
    Only processes tickers in ticker_keys, updates data_dict in-place.

    Returns:
        Only the audit logs (the data_dict is modified in-place)
    """
    total_tickers = len(ticker_keys)
    all_audit_logs = []
    failed_tickers = []

    if batch_size is None:
        batch_size = min(DEFAULT_BATCH_SIZE, total_tickers)

    for batch_start in range(0, total_tickers, batch_size):
        batch_end = min(batch_start + batch_size, total_tickers)
        batch_tickers = ticker_keys[batch_start:batch_end]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {}

            for ticker in batch_tickers:
                payload = data_dict[ticker]

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

            iterator = tqdm(
                as_completed(future_to_ticker.keys()),
                total=len(future_to_ticker),
                disable=not show_progress,
                desc="Sector tickers"
            )

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    cleaned_lf, audit_log, returned_metadata = future.result()

                    # Update in-place
                    data_dict[ticker] = (cleaned_lf, returned_metadata)

                    # Truncate and store log
                    truncated_log = truncate_ticker_logs(audit_log, max_logs_per_ticker)
                    if get_log_size(truncated_log) > 0:
                        all_audit_logs.append(truncated_log)

                    del cleaned_lf, audit_log, returned_metadata

                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
                    failed_tickers.append(ticker)

            future_to_ticker.clear()

        gc.collect()

    return all_audit_logs


# Keep old function for backward compatibility but mark as deprecated
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
    DEPRECATED: Use parallel_process_tickers_inplace instead.

    This function creates a new dictionary which doubles memory usage.
    Kept for backward compatibility only.
    """
    logger.warning("parallel_process_tickers is deprecated - use parallel_process_tickers_inplace instead")

    # Call in-place version (modifies data_dict)
    audit_logs = parallel_process_tickers_inplace(
        data_dict=data_dict,
        columns=columns,
        function=function,
        max_workers=max_workers,
        batch_size=batch_size,
        show_progress=show_progress,
        max_logs_per_ticker=max_logs_per_ticker,
        shared_data=shared_data
    )

    # Return the modified dict (same reference) and logs
    return data_dict, audit_logs
