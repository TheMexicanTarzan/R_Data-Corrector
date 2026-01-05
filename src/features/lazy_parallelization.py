from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import gc

import polars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MEMORY FIX: Limit audit log entries per ticker for fair distribution
MAX_LOGS_PER_TICKER = 100


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
                          function: Callable) -> tuple[polars.LazyFrame, list[dict]]:
    """
    Process a single ticker file through any cleaning function
    """

    # Apply cleaning functions in sequence
    lf, audit = function(df=lf, columns=columns, ticker=ticker)

    return ticker, lf, audit


def parallel_process_tickers(
        data_dict: dict[str, polars.LazyFrame],
        columns: list[str] = [""],
        function: Callable = None,
        max_workers: int = 8,
        batch_size: int = None,
        show_progress: bool = True,
        max_logs_per_ticker: int = MAX_LOGS_PER_TICKER
) -> tuple[dict[str, polars.LazyFrame], list[dict]]:
    """
    Parallelize the processing of multiple tickers using threading.
    Threading works well with Polars since it releases the GIL.

    MEMORY OPTIMIZATIONS:
    - Per-ticker log truncation for fair distribution
    - Explicit garbage collection between batches
    - LazyFrames are kept lazy (not collected)

    Args:
        data_dict: Dictionary mapping ticker symbols to LazyFrames
        columns: List of column names to process
        function: Validation/cleaning function to apply
        max_workers: Number of parallel threads
        batch_size: Number of tickers per batch
        show_progress: Whether to show progress bar
        max_logs_per_ticker: Maximum log entries per ticker (default: MAX_LOGS_PER_TICKER)

    Returns:
        Tuple of (cleaned_lazyframes_dict, all_audit_logs_list)
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
                    f"tickers {batch_start} to {batch_end - 1} ({len(batch_tickers)} tickers)")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(
                    process_single_ticker,
                    data_dict[ticker],
                    columns,
                    ticker,
                    function
                ): ticker for ticker in batch_tickers
            }

            futures = list(future_to_ticker.keys())
            iterator = tqdm(as_completed(futures), total=len(futures),
                            disable=not show_progress,
                            desc=f"Batch {batch_start // batch_size + 1}")

            for future in iterator:
                ticker = future_to_ticker[future]
                try:
                    returned_ticker, cleaned_lf, audit_log = future.result()
                    cleaned_lazyframes[returned_ticker] = cleaned_lf

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
