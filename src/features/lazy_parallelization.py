from typing import Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import logging

import polars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_single_ticker(lf: polars.LazyFrame,
                          columns: list[str],
                          ticker: str,
                          function: Callable) -> tuple[polars.LazyFrame, list[dict]]:
    """
    Process a single ticker file through any cleaning function
    """

    # Apply cleaning functions in sequence
    lf, audit = function(df = lf, columns = columns, ticker = ticker)


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
    """
    total_tickers = len(data_dict)
    cleaned_lazyframes = {}
    all_audit_logs = []
    failed_tickers = []

    if batch_size is None:
        batch_size = total_tickers

    logger.info(f"Starting parallel processing of {total_tickers} tickers with {max_workers} threads")

    for batch_start in range(0, total_tickers, batch_size):
        batch_end = min(batch_start + batch_size, total_tickers)
        batch_tickers = list(data_dict.keys())[batch_start:batch_end]

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
                    all_audit_logs.append(audit_log)
                except Exception as exc:
                    logger.error(f'{ticker} generated an exception: {exc}')
                    failed_tickers.append(ticker)

        logger.info(f"Batch complete. Total processed: {len(cleaned_lazyframes)}/{total_tickers}")

    if failed_tickers:
        logger.warning(f"Failed to process {len(failed_tickers)} tickers: {failed_tickers}")

    return cleaned_lazyframes, all_audit_logs
