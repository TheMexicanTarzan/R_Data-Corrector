"""
Comprehensive Schema Override for Financial Data CSV Files

This schema ensures all columns are parsed with correct data types,
preventing the "cannot parse as dtype i64" error.

Based on the Financial Data Error Detection Framework and column patterns.
"""

import polars
import re
from typing import Dict

# ============================================================================
# COMPLETE SCHEMA OVERRIDE
# ============================================================================

FINANCIAL_DATA_SCHEMA = {
    # ========================================================================
    # MARKET DATA - All prices are Float64, volume is Int64
    # ========================================================================
    # Standard (unadjusted)
    'm_open': polars.Float64,
    'm_high': polars.Float64,
    'm_low': polars.Float64,
    'm_close': polars.Float64,
    'm_volume': polars.Int64,
    'm_vwap': polars.Float64,
    'm_transactions': polars.Int64,

    # Split-adjusted
    'm_open_split_adjusted': polars.Float64,
    'm_high_split_adjusted': polars.Float64,
    'm_low_split_adjusted': polars.Float64,
    'm_close_split_adjusted': polars.Float64,
    'm_volume_split_adjusted': polars.Int64,
    'm_vwap_split_adjusted': polars.Float64,

    # Dividend and split-adjusted
    'm_open_dividend_and_split_adjusted': polars.Float64,
    'm_high_dividend_and_split_adjusted': polars.Float64,
    'm_low_dividend_and_split_adjusted': polars.Float64,
    'm_close_dividend_and_split_adjusted': polars.Float64,
    'm_volume_dividend_and_split_adjusted': polars.Int64,
    'm_vwap_dividend_and_split_adjusted': polars.Float64,

    # Market date
    'm_date': polars.Date,

    # ========================================================================
    # BALANCE SHEET (fbs_*) - All Float64 except counts
    # ========================================================================
    'fbs_cash_and_cash_equivalents': polars.Float64,
    'fbs_cash_and_shortterm_investments': polars.Float64,
    'fbs_net_inventory': polars.Float64,
    'fbs_net_receivables': polars.Float64,
    'fbs_other_current_assets': polars.Float64,
    'fbs_current_assets': polars.Float64,

    'fbs_net_property_plant_and_equipment': polars.Float64,
    'fbs_goodwill': polars.Float64,
    'fbs_net_intangible_assets_excluding_goodwill': polars.Float64,
    'fbs_net_intangible_assets_including_goodwill': polars.Float64,
    'fbs_longterm_investments': polars.Float64,
    'fbs_tax_assets': polars.Float64,
    'fbs_other_noncurrent_assets': polars.Float64,
    'fbs_noncurrent_assets': polars.Float64,

    'fbs_assets': polars.Float64,

    'fbs_accounts_payable': polars.Float64,
    'fbs_current_debt': polars.Float64,
    'fbs_current_tax_payables': polars.Float64,
    'fbs_current_deferred_revenue': polars.Float64,
    'fbs_current_accrued_expenses': polars.Float64,
    'fbs_other_current_liabilities': polars.Float64,
    'fbs_current_liabilities': polars.Float64,

    'fbs_longterm_debt': polars.Float64,
    'fbs_noncurrent_deferred_revenue': polars.Float64,
    'fbs_noncurrent_deferred_taxes_liabilities': polars.Float64,
    'fbs_other_noncurrent_liabilities': polars.Float64,
    'fbs_noncurrent_liabilities': polars.Float64,

    'fbs_liabilities': polars.Float64,

    'fbs_common_stock_value': polars.Float64,
    'fbs_preferred_stock_value': polars.Float64,
    'fbs_additional_paid_in_capital': polars.Float64,
    'fbs_retained_earnings': polars.Float64,
    'fbs_treasury_stock_value': polars.Float64,
    'fbs_accumulated_other_comprehensive_income': polars.Float64,
    'fbs_other_stockholder_equity': polars.Float64,
    'fbs_stockholder_equity': polars.Float64,
    'fbs_noncontrolling_interest': polars.Float64,
    'fbs_total_equity_including_noncontrolling_interest': polars.Float64,

    'fbs_liabilities_and_stockholder_equity': polars.Float64,

    # ========================================================================
    # INCOME STATEMENT (fis_*) - All Float64 except share counts
    # ========================================================================
    'fis_revenues': polars.Float64,
    'fis_cost_of_revenues': polars.Float64,
    'fis_gross_profit': polars.Float64,

    'fis_research_and_development': polars.Float64,
    'fis_selling_general_and_administrative': polars.Float64,
    'fis_other_operating_expenses': polars.Float64,
    'fis_operating_expenses': polars.Float64,

    'fis_operating_income': polars.Float64,

    'fis_interest_expense': polars.Float64,
    'fis_interest_income': polars.Float64,
    'fis_other_nonoperating_income_expense': polars.Float64,
    'fis_nonoperating_income': polars.Float64,

    'fis_pretax_income': polars.Float64,

    'fis_income_tax': polars.Float64,

    'fis_net_income_continuing_operations': polars.Float64,
    'fis_net_income_discontinued_operations': polars.Float64,
    'fis_net_income': polars.Float64,
    'fis_net_income_noncontrolling_interest': polars.Float64,
    'fis_net_income_to_common': polars.Float64,

    'fis_preferred_dividends_impact': polars.Float64,

    'fis_weighted_average_basic_shares_outstanding': polars.Float64,
    'fis_weighted_average_diluted_shares_outstanding': polars.Float64,

    'fis_basic_eps': polars.Float64,
    'fis_diluted_eps': polars.Float64,

    'fis_basic_average_shares': polars.Float64,
    'fis_diluted_average_shares': polars.Float64,

    'fis_ebitda': polars.Float64,
    'fis_ebit': polars.Float64,

    # ========================================================================
    # CASH FLOW STATEMENT (fcf_*) - All Float64
    # ========================================================================
    'fcf_net_income': polars.Float64,

    'fcf_depreciation_amortization': polars.Float64,
    'fcf_stock_based_compensation': polars.Float64,
    'fcf_deferred_income_tax': polars.Float64,
    'fcf_other_noncash_items': polars.Float64,

    'fcf_change_accounts_receivable': polars.Float64,
    'fcf_change_inventory': polars.Float64,
    'fcf_change_accounts_payable': polars.Float64,
    'fcf_change_other': polars.Float64,
    'fcf_change_working_capital': polars.Float64,

    'fcf_net_cash_from_operating_activities': polars.Float64,

    'fcf_capital_expenditure': polars.Float64,
    'fcf_acquisition_net': polars.Float64,
    'fcf_purchase_of_investments': polars.Float64,
    'fcf_sale_maturity_of_investments': polars.Float64,
    'fcf_other_investing_activities': polars.Float64,
    'fcf_net_cash_from_investing_activites': polars.Float64,

    'fcf_debt_repayment': polars.Float64,
    'fcf_common_stock_issued': polars.Float64,
    'fcf_common_stock_repurchased': polars.Float64,
    'fcf_dividends_paid': polars.Float64,
    'fcf_other_financing_activities': polars.Float64,
    'fcf_net_cash_from_financing_activities': polars.Float64,

    'fcf_cash_exchange_rate_effect': polars.Float64,

    'fcf_net_change_in_cash': polars.Float64,

    'fcf_period_start_cash': polars.Float64,
    'fcf_period_end_cash': polars.Float64,

    'fcf_free_cash_flow': polars.Float64,

    # ========================================================================
    # FILING METADATA (f_*) - Mix of dates, strings, and floats
    # ========================================================================
    'f_filing_date': polars.Date,
    'f_period_end_date': polars.Date,
    'f_fiscal_year': polars.Int32,
    'f_fiscal_quarter': polars.Int32,
    'f_fiscal_period': polars.String,
    'f_fiscal_sector': polars.String,
    'f_fiscal_industry': polars.String,
    'f_reported_currency': polars.String,
    'f_cik': polars.String,
    'f_ticker': polars.String,

    # ========================================================================
    # DIVIDEND DATA (d_*) - Amounts are Float64, dates are Date
    # ========================================================================
    'd_declaration_date': polars.Date,
    'd_ex_dividend_date': polars.Date,
    'd_record_date': polars.Date,
    'd_payment_date': polars.Date,
    'd_ex_dividend_date_dividend': polars.Float64,
    'd_frequency': polars.Int32,
    'd_dividend': polars.Float64,
    'd_amount': polars.Float64,

    # ========================================================================
    # SPLIT DATA (s_*) - Ratios are Int64, dates are Date, factors Float64
    # ========================================================================
    's_split_date': polars.Date,
    's_split_date_numerator': polars.Int32,
    's_split_date_denominator': polars.Int32,
    's_split_ratio': polars.Float64,
    's_split_factor': polars.Float64,

    # ========================================================================
    # CALCULATED COLUMNS (c_*) - ALL Float64
    # ========================================================================
    # These are computed/derived columns - always use Float64
    'c_market_cap': polars.Float64,
    'c_enterprise_value': polars.Float64,
    'c_price_to_earnings': polars.Float64,
    'c_price_to_book': polars.Float64,
    'c_price_to_sales': polars.Float64,
    'c_ev_to_ebitda': polars.Float64,
    'c_ev_to_sales': polars.Float64,
    'c_debt_to_equity': polars.Float64,
    'c_current_ratio': polars.Float64,
    'c_quick_ratio': polars.Float64,
    'c_roe': polars.Float64,
    'c_roa': polars.Float64,
    'c_roic': polars.Float64,
    'c_gross_margin': polars.Float64,
    'c_operating_margin': polars.Float64,
    'c_net_margin': polars.Float64,
    'c_asset_turnover': polars.Float64,
    'c_inventory_turnover': polars.Float64,
    'c_days_sales_outstanding': polars.Float64,
    'c_days_inventory_outstanding': polars.Float64,
    'c_days_payables_outstanding': polars.Float64,
    'c_cash_conversion_cycle': polars.Float64,

    # Log transformations
    'c_log_market_cap': polars.Float64,
    'c_log_price': polars.Float64,
    'c_log_volume': polars.Float64,
    'c_log_revenues': polars.Float64,
    'c_log_assets': polars.Float64,

    # Differences and returns
    'c_log_difference_high_to_low': polars.Float64,
    'c_price_return': polars.Float64,
    'c_log_return': polars.Float64,
    'c_daily_return': polars.Float64,
    'c_weekly_return': polars.Float64,
    'c_monthly_return': polars.Float64,
    'c_volume_change': polars.Float64,
    'c_price_change': polars.Float64,
    'c_percentage_change': polars.Float64,

    # Volatility measures
    'c_volatility': polars.Float64,
    'c_volatility_5d': polars.Float64,
    'c_volatility_10d': polars.Float64,
    'c_volatility_20d': polars.Float64,
    'c_volatility_60d': polars.Float64,
    'c_historical_volatility': polars.Float64,
    'c_annualized_volatility_5d_log_returns_dividend_and_split_adjusted': polars.Float64,

    # Moving averages
    'c_sma_5': polars.Float64,
    'c_sma_10': polars.Float64,
    'c_sma_20': polars.Float64,
    'c_sma_50': polars.Float64,
    'c_sma_100': polars.Float64,
    'c_sma_200': polars.Float64,
    'c_ema_5': polars.Float64,
    'c_ema_10': polars.Float64,
    'c_ema_20': polars.Float64,
    'c_ema_50': polars.Float64,

    # Technical indicators
    'c_rsi': polars.Float64,
    'c_rsi_14': polars.Float64,
    'c_macd': polars.Float64,
    'c_macd_signal': polars.Float64,
    'c_macd_histogram': polars.Float64,
    'c_bollinger_upper': polars.Float64,
    'c_bollinger_lower': polars.Float64,
    'c_bollinger_middle': polars.Float64,
    'c_atr': polars.Float64,
    'c_atr_14': polars.Float64,

    # Statistical measures
    'c_mean': polars.Float64,
    'c_median': polars.Float64,
    'c_std': polars.Float64,
    'c_variance': polars.Float64,
    'c_skew': polars.Float64,
    'c_kurtosis': polars.Float64,
    'c_correlation': polars.Float64,
    'c_covariance': polars.Float64,
    'c_beta': polars.Float64,
    'c_alpha': polars.Float64,
    'c_sharpe_ratio': polars.Float64,
    'c_sortino_ratio': polars.Float64,

    # Normalized/scaled values
    'c_normalized_price': polars.Float64,
    'c_normalized_volume': polars.Float64,
    'c_scaled_price': polars.Float64,
    'c_scaled_volume': polars.Float64,
    'c_standardized_return': polars.Float64,
    'c_z_score': polars.Float64,

    # Ratios and percentages
    'c_dividend_yield': polars.Float64,
    'c_payout_ratio': polars.Float64,
    'c_retention_ratio': polars.Float64,
    'c_earnings_growth': polars.Float64,
    'c_revenue_growth': polars.Float64,
    'c_fcf_growth': polars.Float64,
    'c_yoy_growth': polars.Float64,
    'c_qoq_growth': polars.Float64,
    'c_cagr': polars.Float64,

    # Percent changes
    'c_pct_change': polars.Float64,
    'c_pct_change_1d': polars.Float64,
    'c_pct_change_5d': polars.Float64,
    'c_pct_change_20d': polars.Float64,
    'c_pct_from_high': polars.Float64,
    'c_pct_from_low': polars.Float64,

    'c_chaikin_money_flow_21d_dividend_and_split_adjusted': polars.Float64,

    # ========================================================================
    # COMMON IDENTIFIER/METADATA COLUMNS
    # ========================================================================
    'ticker': polars.String,
    'company_name': polars.String,
    'sector': polars.String,
    'industry': polars.String,
    'exchange': polars.String,
    'country': polars.String,
    'currency': polars.String,
    'isin': polars.String,
    'cusip': polars.String,
    'sedol': polars.String,

    # ========================================================================
    # RANKING/COUNTING COLUMNS - Int64
    # ========================================================================
    'c_rank': polars.Int64,
    'c_sector_rank': polars.Int64,
    'c_industry_rank': polars.Int64,
    'c_percentile': polars.Float64,
    'c_count': polars.Int64,
    'c_row_number': polars.Int64,
}


# ============================================================================
# PATTERN-BASED SCHEMA INFERENCE
# ============================================================================

def infer_schema_from_patterns(column_name: str) -> polars.DataType:
    """
    Infers the appropriate Polars dtype based on column name patterns.
    Useful when new calculated columns are added dynamically.

    Args:
        column_name: Name of the column

    Returns:
        Polars dtype

    Examples:
        >>> infer_schema_from_patterns('c_annualized_volatility_5d')
        Float64
        >>> infer_schema_from_patterns('m_date')
        Date
        >>> infer_schema_from_patterns('ticker')
        String
    """
    # Date columns (highest priority - check first)
    if re.match(r'.*_date$', column_name, re.IGNORECASE):
        return polars.Date

    # All calculated columns (c_*) are ALWAYS Float64
    # This is the most important rule for your use case
    if column_name.startswith('c_'):
        # Exception: count/rank columns are Int64
        if any(x in column_name.lower() for x in ['_count', '_rank', '_row_number']):
            return polars.Int64
        # Everything else starting with c_ is Float64
        return polars.Float64

    # Market data columns
    if column_name.startswith('m_'):
        # Volume columns are Int64 (unless they have adjustment ratios or percentages)
        if 'volume' in column_name.lower() and not any(
                x in column_name.lower() for x in ['ratio', 'pct', 'change', 'normalized', 'scaled']):
            return polars.Int64
        # Transaction counts are Int64
        if 'transactions' in column_name.lower():
            return polars.Int64
        # All other market columns (prices, vwap, etc.) are Float64
        return polars.Float64

    # Fundamental columns (fbs_, fis_, fcf_) - all Float64
    if column_name.startswith(('fbs_', 'fis_', 'fcf_')):
        return polars.Float64

    # Dividend columns
    if column_name.startswith('d_'):
        # Frequency is Int32
        if 'frequency' in column_name.lower():
            return polars.Int32
        # All amounts are Float64
        return polars.Float64

    # Split columns
    if column_name.startswith('s_'):
        # Numerator/denominator are Int32
        if 'numerator' in column_name.lower() or 'denominator' in column_name.lower():
            return polars.Int32
        # Ratio and factor are Float64
        return polars.Float64

    # Filing metadata
    if column_name.startswith('f_'):
        # Fiscal year/quarter are Int32
        if any(x in column_name.lower() for x in ['fiscal_year', 'fiscal_quarter']):
            return polars.Int32
        # String fields
        if any(x in column_name.lower() for x in ['period', 'sector', 'industry', 'currency', 'cik', 'ticker']):
            return polars.String
        # Default for filing data
        return polars.Float64

    # String identifier columns
    if any(x in column_name.lower() for x in
           ['ticker', 'name', 'sector', 'industry', 'currency', 'cik', 'isin', 'cusip', 'sedol', 'exchange', 'country']):
        return polars.String

    # Count/rank columns (Int64)
    if any(x in column_name.lower() for x in ['count', 'rank', 'row_number']):
        return polars.Int64

    # Fiscal year/quarter (Int32)
    if any(x in column_name.lower() for x in ['fiscal_year', 'fiscal_quarter', 'frequency']):
        return polars.Int32

    # Default to Float64 for safety (better than Int64 for financial data)
    # This catches any unknown columns and assumes they're numerical
    return polars.Float64


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_schema_for_columns(columns: list[str]) -> Dict[str, polars.DataType]:
    """
    Returns a schema dictionary containing only the specified columns.
    Uses pattern-based inference for columns not in FINANCIAL_DATA_SCHEMA.

    Args:
        columns: List of column names present in your CSV

    Returns:
        Dictionary mapping column names to Polars dtypes

    Examples:
        >>> schema = get_schema_for_columns(['ticker', 'c_volatility', 'm_close'])
        >>> print(schema)
        {'ticker': String, 'c_volatility': Float64, 'm_close': Float64}
    """
    schema = {}
    for col in columns:
        if col in FINANCIAL_DATA_SCHEMA:
            # Use predefined schema if available
            schema[col] = FINANCIAL_DATA_SCHEMA[col]
        else:
            # Fall back to pattern-based inference
            schema[col] = infer_schema_from_patterns(col)
    return schema


def build_complete_schema_from_file(csv_path: str) -> Dict[str, polars.DataType]:
    """
    Build a complete schema by reading the CSV headers and applying
    both the predefined schema and pattern-based inference.

    This is the recommended function to use when reading CSV files with
    potentially unknown columns.

    Args:
        csv_path: Path to the CSV file

    Returns:
        Complete schema dictionary

    Examples:
        >>> schema = build_complete_schema_from_file('data/AAPL.csv')
        >>> df = polars.read_csv('data/AAPL.csv', schema_overrides=schema)
    """
    # Read just the headers
    with open(csv_path, 'r') as f:
        headers = f.readline().strip().split(',')

    return get_schema_for_columns(headers)


def validate_schema(df: polars.DataFrame, strict: bool = False) -> Dict[str, str]:
    """
    Validate that a DataFrame's schema matches expected types.

    Args:
        df: Polars DataFrame to validate
        strict: If True, raises ValueError on mismatches. If False, returns dict of issues.

    Returns:
        Dictionary of column -> issue pairs (empty if no issues)

    Examples:
        >>> df = polars.read_csv('data.csv')
        >>> issues = validate_schema(df)
        >>> if issues:
        >>>     print(f"Schema issues found: {issues}")
    """
    issues = {}

    for col in df.columns:
        actual_type = df[col].dtype
        expected_type = FINANCIAL_DATA_SCHEMA.get(col) or infer_schema_from_patterns(col)

        if actual_type != expected_type:
            issue = f"Expected {expected_type}, got {actual_type}"
            issues[col] = issue

            if strict:
                raise ValueError(f"Schema mismatch in column '{col}': {issue}")

    return issues