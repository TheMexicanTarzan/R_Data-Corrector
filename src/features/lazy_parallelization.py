from typing import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import logging
import gc

import polars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MEMORY FIX: Limit total audit log entries to prevent memory explosion
MAX_TOTAL_AUDIT_LOGS = 10000


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
        show_progress: bool = True
) -> tuple[dict[str, polars.LazyFrame], list[dict]]:
    """
    Parallelize the processing of multiple tickers using threading.
    Threading works well with Polars since it releases the GIL.

    MEMORY OPTIMIZATIONS:
    - Explicit garbage collection between batches
    - Limited audit log accumulation
    - LazyFrames are kept lazy (not collected)
    """
    total_tickers = len(data_dict)
    cleaned_lazyframes = {}
    all_audit_logs = []
    failed_tickers = []
    total_logs_collected = 0

    if batch_size is None:
        batch_size = total_tickers

    logger.info(f"Starting parallel processing of {total_tickers} tickers with {max_workers} threads")

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

                    # MEMORY FIX: Limit total audit logs to prevent memory explosion
                    if total_logs_collected < MAX_TOTAL_AUDIT_LOGS:
                        if isinstance(audit_log, list):
                            remaining_capacity = MAX_TOTAL_AUDIT_LOGS - total_logs_collected
                            truncated_log = audit_log[:remaining_capacity]
                            all_audit_logs.append(truncated_log)
                            total_logs_collected += len(truncated_log)
                        elif isinstance(audit_log, dict):
                            # For dict-type logs (like negative_fundamentals)
                            all_audit_logs.append(audit_log)
                            # Estimate dict log size
                            total_logs_collected += sum(
                                len(v) if isinstance(v, list) else 1
                                for v in audit_log.values()
                            )
                        else:
                            all_audit_logs.append(audit_log)
                            total_logs_collected += 1

                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
                    failed_tickers.append(ticker)

        # MEMORY FIX: Force garbage collection after each batch
        gc.collect()

        logger.info(f"Batch complete. Total processed: {len(cleaned_lazyframes)}/{total_tickers}, "
                    f"Logs collected: {total_logs_collected}")

    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers: {failed_tickers}")

    if total_logs_collected >= MAX_TOTAL_AUDIT_LOGS:
        logger.warning(f"Audit logs truncated at {MAX_TOTAL_AUDIT_LOGS} entries to prevent memory issues")

    return cleaned_lazyframes, all_audit_logs
