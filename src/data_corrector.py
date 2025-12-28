
from pathlib import Path
import logging

from src.input_handlers.csv_reader import read_csv_files_to_polars
from src.modules.errors.sanity_check.sanity_check import (
    sort_dates,
    fill_negatives_fundamentals,
    fill_negatives_market,
    zero_wipeout,
    mkt_cap_scale_error,
    ohlc_integrity,
    validate_financial_equivalencies
)
from src.features.lazy_parallelization import parallel_process_tickers
from src.dashboard.dashboard import FinancialDashboard


current_dir = Path.cwd()
data_directory = current_dir / ".." / "Input" / "Data"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dataframe_dict = read_csv_files_to_polars(data_directory, max_files= 500)

date_cols = [
        "f_filing_date",
        "m_date",
    ]

fundamental_negatives_columns = [
    "fbs_cash_and_cash_equivalents",
    "fbs_cash_and_shortterm_investments",
    "fbs_net_inventory",
    "fbs_net_property_plant_and_equipment",
    "fbs_goodwill",
    "fbs_net_intangible_assets_excluding_goodwill",
    "fbs_net_intangible_assets_including_goodwill",
    "fbs_assets",
    "fbs_current_assets",
    "fbs_noncurrent_assets",
    "fis_weighted_average_basic_shares_outstanding",
    "fis_weighted_average_diluted_shares_outstanding",
    "fis_revenues"
]

market_negatives_columns = columns = [
    "m_open",
    "m_high",
    "m_low",
    "m_close",
    "m_volume",
    "m_vwap",
    "m_open_split_adjusted",
    "m_high_split_adjusted",
    "m_low_split_adjusted",
    "m_close_split_adjusted",
    "m_volume_split_adjusted",
    "m_vwap_split_adjusted",
    "m_open_dividend_and_split_adjusted",
    "m_high_dividend_and_split_adjusted",
    "m_low_dividend_and_split_adjusted",
    "m_close_dividend_and_split_adjusted",
    "m_volume_dividend_and_split_adjusted",
    "m_vwap_dividend_and_split_adjusted"
]

zero_wipeout_columns = [
    "fis_weighted_average_basic_shares_outstanding",
    "fis_weighted_average_diluted_shares_outstanding"
]

shares_outstanding_10x_columns = [
    "fis_weighted_average_basic_shares_outstanding",
    "c_market_cap"
]

ohlc_integrity_columns = [
    "m_open",
    "m_high",
    "m_low",
    "m_close",
    "m_vwap"
]

dataframe_dict_clean_negatives_fundamentals, negative_fundamentals_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = fundamental_negatives_columns,
        function = fill_negatives_fundamentals
    )


dataframe_dict_clean_negatives_market, negative_market_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = market_negatives_columns,
        function = fill_negatives_market,
    )

dataframe_dict_sorted_dates, unsorted_dates_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = date_cols,
        function = sort_dates,
    )



dataframe_dict_clean_zero_wipeout, zero_wipeout_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = zero_wipeout_columns,
        function = zero_wipeout
    )


dataframe_dict_clean_10x_shares_outstanding, shares_outstanding_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = shares_outstanding_10x_columns,
        function = mkt_cap_scale_error
    )


dataframe_dict_clean_ohlc_integrity, ohlc_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        function = ohlc_integrity
    )

dataframe_dict_clean_financial_equivalencies, financial_unequivalencies_logs = parallel_process_tickers(
        data_dict = dataframe_dict,
        columns = date_cols,
        function = validate_financial_equivalencies,
    )



# Collect LazyFrames into DataFrames for dashboard
original_dfs = {}
cleaned_dfs = {}

logger.info("Converting LazyFrames to DataFrames for dashboard visualization")
for ticker in dataframe_dict.keys():
    try:
        # Collect original data
        original_dfs[ticker] = dataframe_dict[ticker].collect()

        # Collect cleaned data (use the final cleaned version)
        if ticker in dataframe_dict_clean_financial_equivalencies:
            cleaned_dfs[ticker] = dataframe_dict_clean_financial_equivalencies[ticker].collect()

        logger.info(f"Collected data for {ticker}: {original_dfs[ticker].height} rows")
    except Exception as e:
        logger.error(f"Failed to collect data for {ticker}: {e}")


# Organize all logs
all_logs = {
    "negative_fundamentals": negative_fundamentals_logs,
    "negative_market": negative_market_logs,
    "zero_wipeout": zero_wipeout_logs,
    "mkt_cap_scale": shares_outstanding_logs,
    "ohlc_integrity": ohlc_logs,
    "financial_equivalencies": financial_unequivalencies_logs,
    "sort_dates": unsorted_dates_logs
}

# Count total errors
total_errors = 0
for category, logs in all_logs.items():
    if isinstance(logs, list):
        for log_item in logs:
            if isinstance(log_item, dict):
                # Handle different log structures
                if "hard_filter_errors" in log_item:
                    total_errors += len(log_item["hard_filter_errors"])
                if "soft_filter_warnings" in log_item:
                    total_errors += len(log_item["soft_filter_warnings"])
                # Check if it's a dict with column keys
                elif any(isinstance(v, list) for v in log_item.values()):
                    for v in log_item.values():
                        if isinstance(v, list):
                            total_errors += len(v)
                else:
                    total_errors += 1
            elif isinstance(log_item, list):
                total_errors += len(log_item)


dashboard = FinancialDashboard(original_dfs, cleaned_dfs, all_logs)


# Run the dashboard
dashboard.run(debug=True, port=8050)

