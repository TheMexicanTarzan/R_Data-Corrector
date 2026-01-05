from pathlib import Path
import logging
import polars
import json

from src.input_handlers.csv_reader import read_csv_files_to_polars
from src.modules.errors.sanity_check.sanity_check import (
    sort_dates,
    fill_negatives_fundamentals,
    fill_negatives_market,
    zero_wipeout,
    mkt_cap_scale_error,
    ohlc_integrity,
    validate_financial_equivalencies,
    validate_market_split_consistency
)
from src.features.lazy_parallelization import parallel_process_tickers, consolidate_audit_logs
from src.dashboard.dashboard import run_dashboard

current_dir = Path.cwd()
data_directory = current_dir / ".." / "Input" / "Data"
sanity_check_output_logs_directory = current_dir / ".." / "Output" / "sanity_check" / "error_logs"
batch_size = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    dataframe_dict = read_csv_files_to_polars(data_directory, max_files=7000)

    # MEMORY FIX: Don't pre-collect all originals - store file paths for on-demand loading
    # The dashboard will load originals lazily when needed for visualization
    original_file_paths = {
        ticker: data_directory / ticker for ticker in dataframe_dict.keys()
    }


    def run_full_sanity_check():
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

        market_negatives_columns = [
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

        market_inconsistencies_columns = [
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
            "s_split_date_numerator",
            "s_split_date_denominator"
        ]

        dataframe_dict_sorted_dates, unsorted_dates_logs = parallel_process_tickers(
            data_dict=dataframe_dict,
            columns=date_cols,
            function=sort_dates,
            batch_size=batch_size
        )

        dataframe_dict_clean_negatives_fundamentals, negative_fundamentals_logs = parallel_process_tickers(
            data_dict=dataframe_dict_sorted_dates,
            columns=fundamental_negatives_columns,
            function=fill_negatives_fundamentals,
            batch_size=batch_size
        )

        dataframe_dict_clean_negatives_market, negative_market_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_negatives_fundamentals,
            columns=market_negatives_columns,
            function=fill_negatives_market,
            batch_size=batch_size
        )

        dataframe_dict_clean_zero_wipeout, zero_wipeout_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_negatives_market,
            columns=zero_wipeout_columns,
            function=zero_wipeout,
            batch_size=batch_size
        )

        dataframe_dict_clean_10x_shares_outstanding, shares_outstanding_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_zero_wipeout,
            columns=shares_outstanding_10x_columns,
            function=mkt_cap_scale_error,
            batch_size=batch_size
        )

        dataframe_dict_clean_ohlc_integrity, ohlc_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_10x_shares_outstanding,
            columns=ohlc_integrity_columns,
            function=ohlc_integrity,
            batch_size=batch_size
        )

        dataframe_dict_clean_financial_equivalencies, financial_unequivalencies_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_ohlc_integrity,
            function=validate_financial_equivalencies,
            batch_size=batch_size
        )

        dataframe_dict_clean_split_consistency, split_inconsistencies_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_financial_equivalencies,
            columns=market_inconsistencies_columns,
            function=validate_market_split_consistency,
            batch_size=batch_size
        )

        # Consolidate audit logs to remove empty entries from parallel processing
        logs_sanity_check = {
            "unsorted_dates_logs": consolidate_audit_logs(unsorted_dates_logs),
            "negative_fundamentals_logs": consolidate_audit_logs(negative_fundamentals_logs),
            "negative_market_logs": consolidate_audit_logs(negative_market_logs),
            "zero_wipeout_logs": consolidate_audit_logs(zero_wipeout_logs),
            "shares_outstanding_logs": consolidate_audit_logs(shares_outstanding_logs),
            "ohlc_logs": consolidate_audit_logs(ohlc_logs),
            "financial_unequivalencies_logs": consolidate_audit_logs(financial_unequivalencies_logs),
            "split_inconsistencies_logs": consolidate_audit_logs(split_inconsistencies_logs)
        }
        return dataframe_dict_clean_split_consistency, logs_sanity_check


    clean_dfs, logs = run_full_sanity_check()

    with open(sanity_check_output_logs_directory / 'logs_sanity_check.json', 'w') as f:
        json.dump(logs, f, indent=4, default=str)

    print("Data cleaning complete. Launching dashboard...")

    # Launch the dashboard with file paths for on-demand loading (memory efficient)
    # run_dashboard(
    #     original_file_paths=original_file_paths,
    #     cleaned_dataframes=clean_dfs,
    #     logs=logs,
    #     debug=True,
    #     port=8050
    # )