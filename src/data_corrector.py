from pathlib import Path
import logging
import json
import gc

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
    rolling_z_score,
    MetadataCache,
    compute_sector_model
)
from src.features import parallel_process_tickers, consolidate_audit_logs
from src.dashboard import launch_dashboard
from src.output_handlers import save_corrected_data

current_dir = Path.cwd()
data_directory = current_dir / ".." / "Input" / "Data"
metadata_path = current_dir / ".." / "Input" / "Universe_Information" / "Universe_Information.csv"
output_logs_directory = current_dir / ".." / "Output"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_full_sanity_check(data: dict, save_data: bool, out_format: str, output_logs_directory, batch_size: int = 512) -> tuple[dict, dict]:
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
        data_dict=data,
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

    # Save corrected data if requested
    if save_data:
        logger.info("Saving corrected data to CSV files...")
        output_data_directory = output_logs_directory / "sanity_check" / "corrected_data"
        saved_files = save_corrected_data(
            clean_data_dict=dataframe_dict_clean_split_consistency,
            output_directory=output_data_directory,
            file_format=out_format,
            create_directory=True,
            overwrite=True
        )
        logger.info(f"Successfully saved {len(saved_files)} corrected data files to {output_data_directory}")

    return dataframe_dict_clean_split_consistency, logs

def _group_tickers_by_sector(data: dict, metadata_cache: MetadataCache) -> dict[str, list[str]]:
    """
    Group tickers by sector for serial processing.

    Args:
        data: Dictionary of ticker -> (LazyFrame, metadata) tuples
        metadata_cache: Initialized metadata cache with sector mappings

    Returns:
        Dictionary mapping sector -> list of tickers in that sector
    """
    sector_to_tickers = {}

    for ticker in data.keys():
        ticker_symbol = ticker.replace(".csv", "") if ticker.endswith(".csv") else ticker
        sector = metadata_cache.get_sector(ticker_symbol)

        if sector is None:
            # Put tickers without sector in a special "Unknown" group
            sector = "Unknown"

        if sector not in sector_to_tickers:
            sector_to_tickers[sector] = []
        sector_to_tickers[sector].append(ticker)

    return sector_to_tickers

def _process_mahalanobis_by_sector(
        data: dict,
        columns: list[str],
        batch_size: int,
        max_workers: int = 8,
        confidence: float = 0.0001,
        date_col: str = "m_date"
) -> tuple[dict, list[dict]]:
    """
    Process Mahalanobis filtering using Sector-Based Hybrid Strategy.

    Strategy:
    1. Group tickers by sector (Serial execution per sector)
    2. For each sector:
       a. Load only that sector's data
       b. Compute sector model once
       c. Process all tickers in that sector in parallel
       d. Clean up memory before moving to next sector

    This avoids loading multiple sector models simultaneously (OOM prevention).

    Args:
        data: Dictionary of ticker -> (LazyFrame, metadata) tuples
        columns: Columns to filter
        batch_size: Batch size for parallel processing within sector
        max_workers: Number of parallel workers per sector
        confidence: Mahalanobis confidence threshold
        date_col: Date column name

    Returns:
        Tuple of (cleaned_data_dict, all_audit_logs)
    """
    # Initialize metadata cache for sector lookups
    metadata_cache = MetadataCache()
    first_ticker_data = next(iter(data.values()))
    if isinstance(first_ticker_data, tuple) and len(first_ticker_data) > 1:
        metadata_lf = first_ticker_data[1]
        if metadata_lf is not None:
            logger.info("Initializing metadata cache for sector grouping...")
            metadata_cache.initialize(metadata_lf)
            logger.info(f"Metadata cache initialized with {len(metadata_cache._symbol_to_sector)} symbols")

    # Group tickers by sector
    logger.info("Grouping tickers by sector...")
    sector_to_tickers = _group_tickers_by_sector(data, metadata_cache)
    logger.info(f"Found {len(sector_to_tickers)} sectors to process")

    # Process each sector serially
    cleaned_data = {}
    all_audit_logs = []

    for sector_idx, (sector, tickers_in_sector) in enumerate(sector_to_tickers.items(), 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Sector {sector_idx}/{len(sector_to_tickers)}: {sector}")
        logger.info(f"Tickers in sector: {len(tickers_in_sector)}")
        logger.info(f"{'='*80}")

        # Extract only this sector's data (minimize memory footprint)
        sector_data = {ticker: data[ticker] for ticker in tickers_in_sector}

        # Prepare shared_data for this sector (contains all sector tickers for peer analysis)
        shared_data_sector = dict(sector_data)
        shared_data_sector["__metadata_cache__"] = metadata_cache
        shared_data_sector["__schema_cache__"] = {}

        # Compute sector model once (this is the expensive operation)
        logger.info(f"Computing sector model for {sector}...")
        peer_symbols = metadata_cache.get_peer_symbols(sector)

        # Get available columns from first ticker in sector
        first_ticker = tickers_in_sector[0]
        first_payload = sector_data[first_ticker]
        first_lf = first_payload[0] if isinstance(first_payload, tuple) else first_payload
        first_schema = set(first_lf.collect_schema().names())
        available_cols = [col for col in columns if col in first_schema]

        if len(available_cols) < 2 or len(peer_symbols) < 2:
            logger.warning(f"Skipping sector {sector}: insufficient columns or peers")
            # Copy data through without filtering
            for ticker in tickers_in_sector:
                cleaned_data[ticker] = sector_data[ticker]
            continue

        # Compute the sector model
        sector_model = compute_sector_model(
            sector=sector,
            peer_symbols=peer_symbols,
            available_cols=available_cols,
            date_col=date_col,
            confidence=confidence,
            shared_data=shared_data_sector,
            schema_cache=shared_data_sector["__schema_cache__"]
        )

        if sector_model is None:
            logger.warning(f"Failed to compute sector model for {sector}")
            # Copy data through without filtering
            for ticker in tickers_in_sector:
                cleaned_data[ticker] = sector_data[ticker]
            continue

        # Inject pre-computed sector model into shared_data
        shared_data_sector["__sector_model__"] = sector_model
        logger.info(f"Sector model computed successfully for {sector}")

        # Process all tickers in this sector in parallel
        logger.info(f"Processing {len(tickers_in_sector)} tickers in parallel for sector {sector}...")
        sector_cleaned, sector_logs = parallel_process_tickers(
            data_dict=sector_data,
            columns=columns,
            function=mahalanobis_filter,
            batch_size=batch_size,
            max_workers=max_workers,
            shared_data=shared_data_sector
        )

        # Merge results
        cleaned_data.update(sector_cleaned)
        all_audit_logs.extend(sector_logs)

        logger.info(f"Sector {sector} complete. Processed {len(sector_cleaned)} tickers.")

        # CRITICAL: Clean up memory before next sector
        del sector_data
        del sector_cleaned
        del shared_data_sector
        del sector_model
        gc.collect()
        logger.info(f"Memory cleaned for sector {sector}")

    logger.info(f"\n{'='*80}")
    logger.info(f"All sectors processed. Total tickers: {len(cleaned_data)}")
    logger.info(f"{'='*80}\n")

    return cleaned_data, all_audit_logs

def run_full_statistical_filter(data: dict, save_data: bool, out_format: str, output_logs_directory, batch_size: int = 512) -> tuple[dict, dict]:
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
        data_dict=data,
        columns=rolling_z_cols,
        function=rolling_z_score,
        batch_size=batch_size
    )

    # SECTOR-BASED HYBRID STRATEGY: Process Mahalanobis filter by sector
    # - Iterate through sectors serially (one at a time)
    # - Within each sector, process tickers in parallel
    # - Clean up memory after each sector to prevent OOM
    logger.info("Starting Sector-Based Mahalanobis filtering...")
    dataframe_dict_clean_mahalanobis, mahalanobis_logs = _process_mahalanobis_by_sector(
        data=dataframe_dict_clean_rolling,
        columns=mahalanobis_cols,
        batch_size=batch_size,
        max_workers=8,
        confidence=0.0001,
        date_col="m_date"
    )
    logger.info("Sector-Based Mahalanobis filtering complete")

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
        "rolling_z_score_filter": consolidate_audit_logs(rolling_z_logs),
        "mahalanobis_filter": consolidate_audit_logs(mahalanobis_logs),
        "garch_residuals_filter": consolidate_audit_logs(garch_logs),
        "mad_filter": consolidate_audit_logs(mad_logs)
    }

    # Save corrected data if requested
    if save_data:
        with open(output_logs_directory / "statistical_filter" / "error_logs" / 'logs_statistical_filter.json', 'w') as f:
            json.dump(logs, f, indent=4, default=str)

        logger.info("Saving corrected data to CSV files...")
        output_data_directory = output_logs_directory / "statistical_filter" / "corrected_data"
        saved_files = save_corrected_data(
            clean_data_dict=dataframe_dict_clean_garch,
            output_directory=output_data_directory,
            file_format=out_format,
            create_directory=True,
            overwrite=True
        )
        logger.info(f"Successfully saved {len(saved_files)} corrected data files to {output_data_directory}")

    return dataframe_dict_clean_garch, logs

def run_full_pipeline(data: dict, save_data: bool, out_format: str, output_logs_directory, batch_size: int = 512):
    clean_lfs, sanity_logs = run_full_sanity_check(data,
                                            save_data = not save_data,
                                            out_format=out_format,
                                            output_logs_directory=output_logs_directory,
                                            batch_size=batch_size)
    clean_lfs, stats_logs = run_full_statistical_filter(clean_lfs,
                                                  save_data = not save_data,
                                                  out_format=out_format,
                                                  output_logs_directory=output_logs_directory,
                                                  batch_size=batch_size)


    if save_data:
        with open(output_logs_directory / "full_pipeline" / "error_logs" / 'logs_sanity_check.json', 'w') as f:
            json.dump(sanity_logs, f, indent=4, default=str)

        with open(output_logs_directory / "full_pipeline" / "error_logs" / 'logs_statistical_filter.json',
                  'w') as f:
            json.dump(stats_logs, f, indent=4, default=str)

        logger.info("Saving corrected data to CSV files...")
        output_data_directory = output_logs_directory / "full_pipeline" / "corrected_data"
        saved_files = save_corrected_data(
            clean_data_dict=clean_lfs,
            output_directory=output_data_directory,
            file_format=out_format,
            create_directory=True,
            overwrite=True
        )

    logger.info(f"Successfully saved {len(saved_files)} corrected data files to {output_data_directory}")

    return clean_lfs, sanity_logs, stats_logs
