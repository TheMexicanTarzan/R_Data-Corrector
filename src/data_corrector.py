from pathlib import Path
import logging
import json

from src.input_handlers import read_csv_files_to_polars
from src.modules.errors.sanity_check import (
    sort_dates,
    fill_negatives_fundamentals,
    fill_negatives_market,
    zero_wipeout,
    mkt_cap_scale_error,
    ohlc_integrity,
    validate_financial_equivalencies,
    validate_market_split_consistency
)
from src.modules.errors.statistical_filter import (
    garch_residuals,
    mahalanobis_filter,
    mad_filter,
    rolling_z_score
)
from src.features import parallel_process_tickers, consolidate_audit_logs
from src.dashboard import run_dashboard

current_dir = Path.cwd()
data_directory = current_dir / ".." / "Input" / "Data"
metadata_path = current_dir / ".." / "Input" / "Universe_Information" / "Universe_Information.csv"
output_logs_directory = current_dir / ".." / "Output"
batch_size = 512
max_files = 20000

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    dataframe_dict = read_csv_files_to_polars(data_directory, metadata_path=metadata_path, max_files=max_files)

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
        logs = {
            "unsorted_dates_logs": consolidate_audit_logs(unsorted_dates_logs),
            "negative_fundamentals_logs": consolidate_audit_logs(negative_fundamentals_logs),
            "negative_market_logs": consolidate_audit_logs(negative_market_logs),
            "zero_wipeout_logs": consolidate_audit_logs(zero_wipeout_logs),
            "shares_outstanding_logs": consolidate_audit_logs(shares_outstanding_logs),
            "ohlc_logs": consolidate_audit_logs(ohlc_logs),
            "financial_unequivalencies_logs": consolidate_audit_logs(financial_unequivalencies_logs),
            "split_inconsistencies_logs": consolidate_audit_logs(split_inconsistencies_logs)
        }

        with open(output_logs_directory / "sanity_check" / "error_logs" / 'logs_sanity_check.json', 'w') as f:
            json.dump(logs, f, indent=4, default=str)

        return dataframe_dict_clean_split_consistency, logs

    def run_full_statistical_filter():
        # 1. Rolling Statistics (20-60 Day Window)
        # Target: Time-Series Trends (Prices & Moving Averages)
        # These check against recent history (Mean +/- StdDev) to handle drift.
        rolling_z_cols = [
            "m_open", "m_high", "m_low", "m_close", "m_vwap",
            "m_open_split_adjusted", "m_high_split_adjusted", "m_low_split_adjusted", "m_close_split_adjusted",
            "m_vwap_split_adjusted",
            "m_open_dividend_and_split_adjusted", "m_high_dividend_and_split_adjusted",
            "m_low_dividend_and_split_adjusted", "m_close_dividend_and_split_adjusted",
            "m_vwap_dividend_and_split_adjusted",
            "c_simple_moving_average_5d_close_dividend_and_split_adjusted",
            "c_simple_moving_average_5d_close_split_adjusted",
            "c_simple_moving_average_21d_close_dividend_and_split_adjusted",
            "c_simple_moving_average_21d_close_split_adjusted",
            "c_simple_moving_average_63d_close_dividend_and_split_adjusted",
            "c_simple_moving_average_63d_close_split_adjusted",
            "c_simple_moving_average_252d_close_dividend_and_split_adjusted",
            "c_simple_moving_average_252d_close_split_adjusted",
            "c_exponential_moving_average_5d_close_dividend_and_split_adjusted",
            "c_exponential_moving_average_5d_close_split_adjusted",
            "c_exponential_moving_average_21d_close_dividend_and_split_adjusted",
            "c_exponential_moving_average_21d_close_split_adjusted",
            "c_exponential_moving_average_63d_close_dividend_and_split_adjusted",
            "c_exponential_moving_average_63d_close_split_adjusted",
            "c_exponential_moving_average_252d_close_dividend_and_split_adjusted",
            "c_exponential_moving_average_252d_close_split_adjusted"
        ]

        # 2. Mahalanobis Distance (Cross-Sectional)
        # Target: Fundamental Relationships & Valuation Ratios
        # These check multivariate correlations (e.g., Assets vs Liabilities vs Revenue).
        mahalanobis_cols = [
            # Balance Sheet (fbs_)
            "fbs_accumulated_other_comprehensive_income_after_tax", "fbs_additional_paid_in_capital", "fbs_assets",
            "fbs_capital_lease_obligations",
            "fbs_cash_and_cash_equivalents", "fbs_cash_and_shortterm_investments", "fbs_common_stock_value",
            "fbs_current_accounts_payable",
            "fbs_current_accounts_receivable_after_doubtful_accounts", "fbs_current_accrued_expenses",
            "fbs_current_assets", "fbs_current_capital_lease_obligations",
            "fbs_current_liabilities", "fbs_current_net_receivables", "fbs_current_tax_payables",
            "fbs_deferred_revenue", "fbs_goodwill",
            "fbs_investments", "fbs_liabilities", "fbs_longterm_debt", "fbs_longterm_investments", "fbs_net_debt",
            "fbs_net_intangible_assets_excluding_goodwill", "fbs_net_intangible_assets_including_goodwill",
            "fbs_net_inventory",
            "fbs_net_property_plant_and_equipment", "fbs_noncontrolling_interest", "fbs_noncurrent_assets",
            "fbs_noncurrent_capital_lease_obligations",
            "fbs_noncurrent_deferred_revenue", "fbs_noncurrent_deferred_tax_assets",
            "fbs_noncurrent_deferred_tax_liabilities", "fbs_noncurrent_liabilities",
            "fbs_other_assets", "fbs_other_current_assets", "fbs_other_current_liabilities", "fbs_other_liabilities",
            "fbs_other_noncurrent_assets",
            "fbs_other_noncurrent_liabilities", "fbs_other_payables", "fbs_other_receivables",
            "fbs_other_stockholder_equity", "fbs_preferred_stock_value",
            "fbs_prepaid_expenses", "fbs_retained_earnings", "fbs_shortterm_debt", "fbs_shortterm_investments",
            "fbs_stockholder_equity",
            "fbs_total_debt_including_capital_lease_obligations", "fbs_total_equity_including_noncontrolling_interest",
            "fbs_total_liabilities_and_equity",
            "fbs_total_payables_current_and_noncurrent", "fbs_treasury_stock_value",

            # Cash Flow (fcf_)
            "fcf_accounts_payable_change", "fcf_accounts_receivable_change", "fcf_capital_expenditure",
            "fcf_cash_and_cash_equivalents_change",
            "fcf_cash_exchange_rate_effect", "fcf_common_stock_dividend_payments", "fcf_common_stock_issuance_proceeds",
            "fcf_common_stock_repurchase",
            "fcf_deferred_income_tax", "fcf_depreciation_and_amortization", "fcf_dividend_payments",
            "fcf_free_cash_flow", "fcf_interest_payments",
            "fcf_inventory_change", "fcf_investment_sales_maturities_and_collections_proceeds",
            "fcf_investments_purchase", "fcf_net_business_acquisition_payments",
            "fcf_net_cash_from_operating_activities", "fcf_net_cash_from_investing_activites",
            "fcf_net_cash_from_financing_activities",
            "fcf_net_common_stock_issuance_proceeds", "fcf_net_debt_issuance_proceeds", "fcf_net_income",
            "fcf_net_income_tax_payments",
            "fcf_net_longterm_debt_issuance_proceeds", "fcf_net_shortterm_debt_issuance_proceeds",
            "fcf_net_stock_issuance_proceeds",
            "fcf_other_financing_activities", "fcf_other_investing_activities", "fcf_other_noncash_items",
            "fcf_other_working_capital",
            "fcf_period_end_cash", "fcf_period_start_cash", "fcf_preferred_stock_dividend_payments",
            "fcf_preferred_stock_issuance_proceeds",
            "fcf_property_plant_and_equipment_purchase", "fcf_stock_based_compensation", "fcf_working_capital_change",

            # Income Statement (fis_)
            "fis_basic_earnings_per_share", "fis_basic_net_income_available_to_common_stockholders",
            "fis_continuing_operations_income_after_tax",
            "fis_costs_and_expenses", "fis_cost_of_revenue", "fis_depreciation_and_amortization",
            "fis_diluted_earnings_per_share",
            "fis_discontinued_operations_income_after_tax", "fis_earnings_before_interest_and_tax",
            "fis_earnings_before_interest_tax_depreciation_and_amortization",
            "fis_general_and_administrative_expense", "fis_gross_profit", "fis_income_before_tax",
            "fis_income_tax_expense", "fis_interest_expense",
            "fis_interest_income", "fis_net_income", "fis_net_income_deductions", "fis_net_interest_income",
            "fis_net_total_other_income",
            "fis_nonoperating_income_excluding_interest", "fis_operating_expenses", "fis_operating_income",
            "fis_other_expenses",
            "fis_other_net_income_adjustments", "fis_research_and_development_expense", "fis_revenues",
            "fis_selling_and_marketing_expense",
            "fis_selling_general_and_administrative_expense", "fis_weighted_average_basic_shares_outstanding",
            "fis_weighted_average_diluted_shares_outstanding",

            # Calculated Valuation Ratios (c_)
            "c_book_to_price", "c_book_value_per_share", "c_earnings_per_share", "c_earnings_to_price",
            "c_last_twelve_months_net_income", "c_last_twelve_months_revenue", "c_last_twelve_months_revenue_per_share",
            "c_market_cap", "c_sales_to_price"
        ]

        # 3. Modified Z-Score (MAD Filter)
        # Target: Spiky Data & Univariate Outliers
        # Handles volume spikes and bounded oscillators that may break standard deviation.
        mad_cols = [
            # Volume
            "m_volume", "m_volume_split_adjusted", "m_volume_dividend_and_split_adjusted",
            "c_daily_traded_value",
            "c_daily_traded_value_sma_5d", "c_daily_traded_value_sma_21d",
            "c_daily_traded_value_sma_63d", "c_daily_traded_value_sma_252d",

            # Oscillators & Indicators
            "c_chaikin_money_flow_21d_dividend_and_split_adjusted", "c_chaikin_money_flow_21d_split_adjusted",
            "c_macd_26d_12d_dividend_and_split_adjusted", "c_macd_26d_12d_split_adjusted",
            "c_macd_signal_9d_dividend_and_split_adjusted", "c_macd_signal_9d_split_adjusted",
            "c_rsi_14d_dividend_and_split_adjusted", "c_rsi_14d_split_adjusted",

            # Dividend Amounts (Magnitude Checks)
            "d_declaration_date_dividend", "d_declaration_date_dividend_split_adjusted",
            "d_ex_dividend_date_dividend", "d_ex_dividend_date_dividend_split_adjusted",
            "d_record_date_dividend", "d_record_date_dividend_split_adjusted",
            "d_payment_date_dividend", "d_payment_date_dividend_split_adjusted"
        ]

        # 4. GARCH Residuals
        # Target: Volatility & Returns
        # Checks for 'surprise' volatility given the recent market context.
        garch_cols = [
            "c_annualized_volatility_5d_log_returns_dividend_and_split_adjusted",
            "c_annualized_volatility_21d_log_returns_dividend_and_split_adjusted",
            "c_annualized_volatility_63d_log_returns_dividend_and_split_adjusted",
            "c_annualized_volatility_252d_log_returns_dividend_and_split_adjusted",
            "c_log_difference_high_to_low",
            "c_log_returns_dividend_and_split_adjusted"
        ]

        dataframe_dict_clean_rolling, rolling_z_logs = parallel_process_tickers(
            data_dict=dataframe_dict,
            columns=rolling_z_cols,
            function=rolling_z_score,
            batch_size=batch_size
        )

        dataframe_dict_clean_mahalanobis, mahalanobis_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_rolling,
            columns=mahalanobis_cols,
            function=mahalanobis_filter,
            batch_size=batch_size,
            shared_data=dataframe_dict_clean_rolling  # Pass all ticker data for cross-sectional peer analysis
        )

        dataframe_dict_clean_mad, mad_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_mahalanobis,
            columns=mad_cols,
            function=mad_filter,
            batch_size=batch_size
        )

        dataframe_dict_clean_garch, garch_logs = parallel_process_tickers(
            data_dict=dataframe_dict_clean_mad,
            columns=garch_cols,
            function=garch_residuals,
            batch_size=batch_size,
            max_workers=1 # Arch library already uses multithreading
        )

        logs ={
            "rolling_z_score_filter": rolling_z_logs,
            "mahalanobis_filter": mahalanobis_logs,
            "garch_residuals_filter": garch_logs,
            "mad_filter": mad_logs
        }

        with open(output_logs_directory / "statistical_filter" / "error_logs" / 'logs_statistical_filter.json', 'w') as f:
            json.dump(logs, f, indent=4, default=str)

        return dataframe_dict_clean_garch, logs


    clean_lfs, logs = run_full_sanity_check()

    # clean_lfs, logs = run_full_statistical_filter()


    print("Data cleaning complete. Launching dashboard...")

    # Launch the dashboard with file paths for on-demand loading (memory efficient)
    # run_dashboard(
    #     original_file_paths=original_file_paths,
    #     cleaned_dataframes=clean_lfs,
    #     logs=logs,
    #     debug=True,
    #     port=8050
    # )