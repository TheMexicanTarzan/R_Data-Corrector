"""
Financial Data Cleaning Dashboard

A robust, interactive Dash application to visualize the results of the financial
data cleaning pipeline. This dashboard serves as an audit tool for comparing
original vs. cleaned data and inspecting error logs.
"""

import polars as pl
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path

import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# LOG NORMALIZER - Unifies different log structures into a standard format
# =============================================================================

class LogNormalizer:
    """
    Normalizes different log structures from various cleaning functions
    into a unified Polars DataFrame for display and analysis.
    """

    @staticmethod
    def normalize_logs(
        all_logs: Dict[str, Any],
        original_dfs: Dict[str, pl.DataFrame],
        cleaned_dfs: Dict[str, pl.DataFrame]
    ) -> pl.DataFrame:
        """
        Convert all log structures into a unified DataFrame.

        Args:
            all_logs: Dictionary mapping error categories to their logs
            original_dfs: Dictionary of original dataframes by ticker
            cleaned_dfs: Dictionary of cleaned dataframes by ticker

        Returns:
            Unified Polars DataFrame with standardized columns
        """
        print("Normalizing logs for dashboard...")
        normalized_rows = []

        # Process each category with progress indication
        categories = [
            ("negative_fundamentals", LogNormalizer._normalize_negative_fundamentals),
            ("negative_market", LogNormalizer._normalize_negative_market),
            ("zero_wipeout", LogNormalizer._normalize_zero_wipeout),
            ("mkt_cap_scale", LogNormalizer._normalize_mkt_cap_scale),
            ("ohlc_integrity", LogNormalizer._normalize_ohlc_integrity),
            ("financial_equivalencies", LogNormalizer._normalize_financial_equivalencies),
            ("sort_dates", LogNormalizer._normalize_sort_dates)
        ]

        for category_name, normalizer_func in categories:
            if category_name in all_logs and all_logs[category_name]:
                print(f"  Processing {category_name}...", end=" ")
                try:
                    rows = normalizer_func(all_logs[category_name])
                    normalized_rows.extend(rows)
                    print(f"✓ ({len(rows)} entries)")
                except Exception as e:
                    print(f"✗ Error: {e}")
                    # Continue processing other categories
                    continue

        # Define explicit schema for type safety and performance
        schema = {
            "ticker": pl.Utf8,
            "date": pl.Utf8,
            "error_category": pl.Utf8,
            "error_type": pl.Utf8,
            "column": pl.Utf8,
            "original_value": pl.Float64,
            "corrected_value": pl.Float64,
            "message": pl.Utf8,
            "severity": pl.Utf8,
            "false_positive": pl.Boolean
        }

        if not normalized_rows:
            # Return empty DataFrame with schema
            return pl.DataFrame(schema=schema)

        # Ensure all rows have consistent types before creating DataFrame
        for row in normalized_rows:
            # Ensure float fields are properly typed (convert None to float NaN for consistency)
            if row.get("original_value") is None:
                row["original_value"] = float('nan')
            elif not isinstance(row["original_value"], (int, float)):
                try:
                    row["original_value"] = float(row["original_value"])
                except (ValueError, TypeError):
                    row["original_value"] = float('nan')

            if row.get("corrected_value") is None:
                row["corrected_value"] = float('nan')
            elif not isinstance(row["corrected_value"], (int, float)):
                try:
                    row["corrected_value"] = float(row["corrected_value"])
                except (ValueError, TypeError):
                    row["corrected_value"] = float('nan')

            # Ensure string fields are strings
            for field in ["ticker", "date", "error_category", "error_type", "column", "message", "severity"]:
                if row.get(field) is None:
                    row[field] = ""
                else:
                    row[field] = str(row[field])

            # Ensure boolean field
            if "false_positive" not in row:
                row["false_positive"] = False

        # Create DataFrame with explicit schema for better performance
        df = pl.DataFrame(normalized_rows, schema=schema, infer_schema_length=None)

        return df

    @staticmethod
    def _normalize_negative_fundamentals(logs: List[Dict]) -> List[Dict]:
        """Normalize fill_negatives_fundamentals logs (dict of lists)."""
        rows = []
        for log_dict in logs:
            if isinstance(log_dict, dict):
                for column, entries in log_dict.items():
                    if isinstance(entries, list):
                        for entry in entries:
                            # Safely extract and convert values
                            try:
                                orig_val = float(entry.get(column, 0))
                            except (ValueError, TypeError):
                                orig_val = float('nan')

                            rows.append({
                                "ticker": str(entry.get("ticker", "")),
                                "date": str(entry.get("m_date", "")),
                                "error_category": "Negative Fundamentals",
                                "error_type": "negative_value",
                                "column": str(column),
                                "original_value": orig_val,
                                "corrected_value": float('nan'),  # Forward filled - actual value unknown
                                "message": f"Negative value in {column} replaced with forward fill",
                                "severity": "warning",
                                "false_positive": False
                            })
        return rows

    @staticmethod
    def _normalize_negative_market(logs: List[Dict]) -> List[Dict]:
        """Normalize fill_negatives_market logs (list of dicts)."""
        rows = []
        for log_list in logs:
            if isinstance(log_list, list):
                for entry in log_list:
                    # Safely convert values
                    try:
                        orig_val = float(entry.get("original_value", float('nan')))
                    except (ValueError, TypeError):
                        orig_val = float('nan')

                    try:
                        corr_val = float(entry.get("corrected_value", float('nan')))
                    except (ValueError, TypeError):
                        corr_val = float('nan')

                    rows.append({
                        "ticker": str(entry.get("ticker", "")),
                        "date": str(entry.get("date", "")),
                        "error_category": "Negative Market Data",
                        "error_type": "negative_value",
                        "column": str(entry.get("column", "")),
                        "original_value": orig_val,
                        "corrected_value": corr_val,
                        "message": f"Method: {entry.get('method', 'unknown')}",
                        "severity": "warning",
                        "false_positive": False
                    })
        return rows

    @staticmethod
    def _normalize_zero_wipeout(logs: List[Dict]) -> List[Dict]:
        """Normalize zero_wipeout logs (list of dicts)."""
        rows = []
        for log_list in logs:
            if isinstance(log_list, list):
                for entry in log_list:
                    ticker = str(entry.get("ticker", ""))
                    date = str(entry.get("m_date", ""))

                    try:
                        volume = float(entry.get("m_volume", 0))
                    except (ValueError, TypeError):
                        volume = 0.0

                    # Multiple columns might be zero in same row
                    for key, value in entry.items():
                        if key not in ["ticker", "m_date", "m_volume"] and value == 0:
                            rows.append({
                                "ticker": ticker,
                                "date": date,
                                "error_category": "Zero Wipeout",
                                "error_type": "zero_with_volume",
                                "column": str(key),
                                "original_value": 0.0,
                                "corrected_value": float('nan'),  # Forward filled - actual value unknown
                                "message": f"Zero value with volume={volume:.0f}",
                                "severity": "error",
                                "false_positive": False
                            })
        return rows

    @staticmethod
    def _normalize_mkt_cap_scale(logs: List[Dict]) -> List[Dict]:
        """Normalize mkt_cap_scale_error logs (list of dicts)."""
        rows = []
        for log_list in logs:
            if isinstance(log_list, list):
                for entry in log_list:
                    # Extract the column that had the error (excluding ticker, date, error_type)
                    ticker = str(entry.get("ticker", ""))
                    date = str(entry.get("m_date", ""))

                    for key, value in entry.items():
                        if key not in ["ticker", "m_date", "error_type"]:
                            # Safely convert value
                            try:
                                orig_val = float(value)
                            except (ValueError, TypeError):
                                orig_val = float('nan')

                            rows.append({
                                "ticker": ticker,
                                "date": date,
                                "error_category": "Market Cap Scale",
                                "error_type": str(entry.get("error_type", "scale_error")),
                                "column": str(key),
                                "original_value": orig_val,
                                "corrected_value": float('nan'),  # Forward filled - actual value unknown
                                "message": "10x jump detected and corrected",
                                "severity": "error",
                                "false_positive": False
                            })
        return rows

    @staticmethod
    def _normalize_ohlc_integrity(logs: List[Dict]) -> List[Dict]:
        """Normalize ohlc_integrity logs (list of dicts)."""
        rows = []
        for log_entry in logs:
            # Handle both list of lists and single list formats
            if isinstance(log_entry, list):
                # It's a list (possibly nested)
                for item in log_entry:
                    if isinstance(item, list):
                        # Nested list
                        for entry in item:
                            if isinstance(entry, dict):
                                rows.extend(LogNormalizer._process_ohlc_entry(entry))
                    elif isinstance(item, dict):
                        # Direct dict
                        rows.extend(LogNormalizer._process_ohlc_entry(item))
            elif isinstance(log_entry, dict):
                # Direct dict
                rows.extend(LogNormalizer._process_ohlc_entry(log_entry))
        return rows

    @staticmethod
    def _process_ohlc_entry(entry: Dict) -> List[Dict]:
        """Process a single OHLC integrity log entry."""
        rows = []
        if not isinstance(entry, dict):
            return rows

        try:
            error_type = str(entry.get("error_type", ""))
            column_group = str(entry.get("column_group", ""))

            # Determine which column was corrected
            if error_type == "high_not_maximum":
                column = f"m_high_{column_group}" if column_group != "raw" else "m_high"
                old_val = entry.get("old_high")
                new_val = entry.get("new_high")
            elif error_type == "low_not_minimum":
                column = f"m_low_{column_group}" if column_group != "raw" else "m_low"
                old_val = entry.get("old_low")
                new_val = entry.get("new_low")
            elif error_type == "vwap_outside_range":
                column = f"m_vwap_{column_group}" if column_group != "raw" else "m_vwap"
                old_val = entry.get("old_vwap")
                new_val = entry.get("new_vwap")
            else:
                return rows  # Skip unknown error types

            # Safely convert values to float
            try:
                old_val = float(old_val) if old_val is not None else float('nan')
            except (ValueError, TypeError):
                old_val = float('nan')

            try:
                new_val = float(new_val) if new_val is not None else float('nan')
            except (ValueError, TypeError):
                new_val = float('nan')

            rows.append({
                "ticker": str(entry.get("ticker", "")),
                "date": str(entry.get("date", "")),
                "error_category": "OHLC Integrity",
                "error_type": error_type,
                "column": str(column),
                "original_value": old_val,
                "corrected_value": new_val,
                "message": str(entry.get("message", "")),
                "severity": "error",
                "false_positive": False
            })
        except Exception as e:
            # Skip malformed entries
            pass

        return rows

    @staticmethod
    def _normalize_financial_equivalencies(logs: List[Dict]) -> List[Dict]:
        """Normalize validate_financial_equivalencies logs (dict with hard/soft keys)."""
        rows = []
        for log_dict in logs:
            if not isinstance(log_dict, dict):
                continue

            # Process hard filter errors
            hard_errors = log_dict.get("hard_filter_errors", [])
            for entry in hard_errors:
                error_type = str(entry.get("error_type", ""))

                # Determine the columns involved
                if "assets" in error_type:
                    cols_involved = "fbs_current_assets, fbs_noncurrent_assets"
                elif "liabilities" in error_type:
                    cols_involved = "fbs_current_liabilities, fbs_noncurrent_liabilities"
                else:
                    cols_involved = "various"

                # Safely convert difference to float
                try:
                    diff_val = float(entry.get("difference", 0))
                except (ValueError, TypeError):
                    diff_val = float('nan')

                rows.append({
                    "ticker": str(entry.get("ticker", "")),
                    "date": str(entry.get("date", "")),
                    "error_category": "Financial Equivalencies (Hard)",
                    "error_type": error_type,
                    "column": cols_involved,
                    "original_value": diff_val,
                    "corrected_value": 0.0,  # After correction, difference is 0
                    "message": f"Method: {entry.get('correction_method', 'unknown')}",
                    "severity": "error",
                    "false_positive": False
                })

            # Process soft filter warnings
            soft_warnings = log_dict.get("soft_filter_warnings", [])
            for entry in soft_warnings:
                error_type = str(entry.get("error_type", ""))

                # Determine the columns involved
                if "equity" in error_type:
                    cols_involved = "fbs_stockholder_equity components"
                elif "cash" in error_type:
                    cols_involved = "fcf_period_end_cash, fbs_cash_and_cash_equivalents"
                elif "accounting_equation" in error_type:
                    cols_involved = "fbs_assets vs (fbs_liabilities + fbs_stockholder_equity + fbs_noncontrolling_interest)"
                else:
                    cols_involved = "various"

                # Safely convert difference to float
                try:
                    diff_val = float(entry.get("difference", 0))
                except (ValueError, TypeError):
                    diff_val = float('nan')

                rows.append({
                    "ticker": str(entry.get("ticker", "")),
                    "date": str(entry.get("date", "")),
                    "error_category": "Financial Equivalencies (Soft)",
                    "error_type": error_type,
                    "column": cols_involved,
                    "original_value": diff_val,
                    "corrected_value": float('nan'),  # Soft warnings don't correct
                    "message": "Warning - not corrected",
                    "severity": "warning",
                    "false_positive": False
                })

        return rows

    @staticmethod
    def _normalize_sort_dates(logs: List[Dict]) -> List[Dict]:
        """Normalize sort_dates logs (list of dicts)."""
        rows = []
        for log_list in logs:
            if isinstance(log_list, list):
                for entry in log_list:
                    error_type = str(entry.get("error_type", ""))

                    if error_type == "order_mismatch":
                        # Safely convert positions to float
                        try:
                            orig_pos = float(entry.get("original_position", 0))
                        except (ValueError, TypeError):
                            orig_pos = 0.0

                        try:
                            sorted_pos = float(entry.get("sorted_position", 0))
                        except (ValueError, TypeError):
                            sorted_pos = 0.0

                        rows.append({
                            "ticker": str(entry.get("ticker", "")),
                            "date": str(entry.get("m_date", "")),
                            "error_category": "Date Sorting",
                            "error_type": "order_mismatch",
                            "column": "dates",
                            "original_value": orig_pos,
                            "corrected_value": sorted_pos,
                            "message": f"Row moved from position {int(orig_pos)} to {int(sorted_pos)}",
                            "severity": "info",
                            "false_positive": False
                        })
                    elif error_type == "duplicates_removed":
                        # Safely convert count to float
                        try:
                            dup_count = float(entry.get("duplicates_removed", 0))
                        except (ValueError, TypeError):
                            dup_count = 0.0

                        rows.append({
                            "ticker": str(entry.get("ticker", "")),
                            "date": "",
                            "error_category": "Date Sorting",
                            "error_type": "duplicates_removed",
                            "column": "dates",
                            "original_value": dup_count,
                            "corrected_value": float('nan'),
                            "message": f"Strategy: {entry.get('strategy', 'unknown')}",
                            "severity": "info",
                            "false_positive": False
                        })

        return rows


# =============================================================================
# DUMMY DATA GENERATOR - Creates fake financial data for testing
# =============================================================================

class DummyDataGenerator:
    """Generates fake financial data and runs it through cleaning functions."""

    @staticmethod
    def generate_ticker_data(
        ticker: str,
        start_date: str = "2022-01-01",
        end_date: str = "2023-12-31",
        inject_errors: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Generate fake financial data for a single ticker.

        Args:
            ticker: Ticker symbol
            start_date: Start date for data
            end_date: End date for data
            inject_errors: Whether to inject errors for testing

        Returns:
            Tuple of (original_df, cleaned_df, logs_dict)
        """
        # Generate date range
        dates = pl.date_range(
            start=datetime.strptime(start_date, "%Y-%m-%d"),
            end=datetime.strptime(end_date, "%Y-%m-%d"),
            interval="1d",
            eager=True
        )

        n_rows = len(dates)

        # Generate base OHLC data
        base_price = 100 + np.random.randn() * 20
        returns = np.random.randn(n_rows) * 0.02
        prices = base_price * np.exp(np.cumsum(returns))

        # Generate OHLC with proper relationships
        close_prices = prices
        open_prices = close_prices * (1 + np.random.randn(n_rows) * 0.01)
        high_prices = np.maximum(open_prices, close_prices) * (1 + np.abs(np.random.randn(n_rows)) * 0.02)
        low_prices = np.minimum(open_prices, close_prices) * (1 - np.abs(np.random.randn(n_rows)) * 0.02)
        vwap = (open_prices + high_prices + low_prices + close_prices) / 4
        volume = np.random.randint(1000000, 10000000, n_rows)

        # Generate fundamental data
        assets = 1_000_000_000 + np.random.randn(n_rows) * 100_000_000
        assets = np.maximum(assets, 100_000_000)  # Keep positive

        current_assets = assets * 0.4
        noncurrent_assets = assets * 0.6

        liabilities = assets * 0.6
        current_liabilities = liabilities * 0.5
        noncurrent_liabilities = liabilities * 0.5

        equity = assets - liabilities
        cash = current_assets * 0.2

        shares_outstanding = 1_000_000_000 + np.random.randn(n_rows) * 10_000_000
        shares_outstanding = np.maximum(shares_outstanding, 100_000_000)

        market_cap = prices * shares_outstanding

        # Create DataFrame
        df = pl.DataFrame({
            "ticker": [ticker] * n_rows,
            "m_date": dates,
            "f_filing_date": dates,
            "f_accepted_date": dates,
            # Market data
            "m_open": open_prices,
            "m_high": high_prices,
            "m_low": low_prices,
            "m_close": close_prices,
            "m_vwap": vwap,
            "m_volume": volume,
            # Split adjusted (same for simplicity)
            "m_open_split_adjusted": open_prices,
            "m_high_split_adjusted": high_prices,
            "m_low_split_adjusted": low_prices,
            "m_close_split_adjusted": close_prices,
            "m_vwap_split_adjusted": vwap,
            # Fundamentals
            "fbs_assets": assets,
            "fbs_current_assets": current_assets,
            "fbs_noncurrent_assets": noncurrent_assets,
            "fbs_liabilities": liabilities,
            "fbs_current_liabilities": current_liabilities,
            "fbs_noncurrent_liabilities": noncurrent_liabilities,
            "fbs_stockholder_equity": equity,
            "fbs_noncontrolling_interest": np.zeros(n_rows),
            "fbs_cash_and_cash_equivalents": cash,
            "fcf_period_end_cash": cash,
            "fis_weighted_average_basic_shares_outstanding": shares_outstanding,
            "fis_weighted_average_diluted_shares_outstanding": shares_outstanding * 1.05,
            "c_market_cap": market_cap,
        })

        # Inject errors if requested
        if inject_errors:
            df = DummyDataGenerator._inject_errors(df)

        original_df = df.clone()

        # Run through cleaning (simplified - just for demo)
        cleaned_df, logs = DummyDataGenerator._simple_clean(df, ticker)

        return original_df, cleaned_df, logs

    @staticmethod
    def _inject_errors(df: pl.DataFrame) -> pl.DataFrame:
        """Inject various types of errors into the data."""
        df_list = df.to_dicts()
        n_rows = len(df_list)

        # Inject negative fundamentals (5% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 20), replace=False):
            df_list[i]["fbs_cash_and_cash_equivalents"] = -abs(df_list[i]["fbs_cash_and_cash_equivalents"])

        # Inject negative market data (3% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 33), replace=False):
            df_list[i]["m_close"] = -abs(df_list[i]["m_close"])

        # Inject OHLC violations (2% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 50), replace=False):
            # Make high lower than close
            df_list[i]["m_high"] = df_list[i]["m_close"] * 0.95

        # Inject zero wipeout (1% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 100), replace=False):
            if df_list[i]["m_volume"] > 0:
                df_list[i]["fis_weighted_average_basic_shares_outstanding"] = 0

        # Inject 10x scale error (0.5% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 200), replace=False):
            df_list[i]["fis_weighted_average_diluted_shares_outstanding"] *= 10

        # Inject accounting mismatch (2% of rows)
        for i in np.random.choice(n_rows, size=max(1, n_rows // 50), replace=False):
            df_list[i]["fbs_current_assets"] *= 1.3  # Break the assets = current + noncurrent identity

        return pl.DataFrame(df_list)

    @staticmethod
    def _simple_clean(df: pl.DataFrame, ticker: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """
        Simple cleaning simulation (generates logs without full cleaning logic).
        In production, this would call the actual cleaning functions.
        """
        logs = {
            "negative_fundamentals": [],
            "negative_market": [],
            "zero_wipeout": [],
            "mkt_cap_scale": [],
            "ohlc_integrity": [],
            "financial_equivalencies": [],
            "sort_dates": []
        }

        # Simulate finding errors and logging them
        df_dicts = df.to_dicts()

        for i, row in enumerate(df_dicts):
            # Check for negative cash
            if row["fbs_cash_and_cash_equivalents"] < 0:
                if not logs["negative_fundamentals"]:
                    logs["negative_fundamentals"].append({})
                if "fbs_cash_and_cash_equivalents" not in logs["negative_fundamentals"][0]:
                    logs["negative_fundamentals"][0]["fbs_cash_and_cash_equivalents"] = []

                logs["negative_fundamentals"][0]["fbs_cash_and_cash_equivalents"].append({
                    "ticker": ticker,
                    "m_date": str(row["m_date"]),
                    "fbs_cash_and_cash_equivalents": row["fbs_cash_and_cash_equivalents"]
                })
                # Fix it
                df_dicts[i]["fbs_cash_and_cash_equivalents"] = abs(row["fbs_cash_and_cash_equivalents"])

            # Check for OHLC violations
            if row["m_high"] < row["m_close"]:
                if not logs["ohlc_integrity"]:
                    logs["ohlc_integrity"].append([])

                logs["ohlc_integrity"][0].append({
                    "ticker": ticker,
                    "date": str(row["m_date"]),
                    "error_type": "high_not_maximum",
                    "column_group": "raw",
                    "old_high": row["m_high"],
                    "new_high": row["m_close"],
                    "message": f"High corrected from {row['m_high']} to {row['m_close']}"
                })
                # Fix it
                df_dicts[i]["m_high"] = max(row["m_open"], row["m_high"], row["m_low"], row["m_close"])

        cleaned_df = pl.DataFrame(df_dicts)

        return cleaned_df, logs


# =============================================================================
# DASHBOARD APPLICATION
# =============================================================================

class FinancialDashboard:
    """Main dashboard application class."""

    def __init__(self, original_dfs: Dict[str, pl.LazyFrame],
                 cleaned_dfs: Dict[str, pl.LazyFrame],
                 all_logs: Dict[str, Any]):
        """
        Initialize the dashboard with memory-efficient lazy evaluation.

        Args:
            original_dfs: Dictionary of original LazyFrames by ticker
            cleaned_dfs: Dictionary of cleaned LazyFrames by ticker
            all_logs: Dictionary of all cleaning logs by category
        """
        # Store LazyFrames (not collected - saves memory!)
        self.original_dfs = original_dfs
        self.cleaned_dfs = cleaned_dfs
        self.all_logs = all_logs

        # Normalize logs (only once, but efficiently)
        print("Initializing dashboard with lazy evaluation for memory efficiency...")
        self.normalized_logs = LogNormalizer.normalize_logs(
            all_logs, {}, {}  # Don't need DataFrames for log normalization
        )

        # Limit normalized logs to prevent memory issues
        if self.normalized_logs.height > 100_000:
            print(f"⚠ Large number of errors ({self.normalized_logs.height:,}). Sampling for performance...")
            # Keep all unique tickers but limit errors per ticker
            self.normalized_logs = self._sample_logs_efficiently(self.normalized_logs)

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            suppress_callback_exceptions=True
        )

        # Store for false positive flags
        self.false_positive_store_path = Path("false_positives.json")
        self.load_false_positives()

        self._build_layout()
        self._register_callbacks()

    def _sample_logs_efficiently(self, logs_df: pl.DataFrame, max_per_ticker: int = 1000) -> pl.DataFrame:
        """
        Sample logs to reduce memory usage while keeping all tickers represented.

        Args:
            logs_df: Full logs DataFrame
            max_per_ticker: Maximum errors per ticker to keep

        Returns:
            Sampled DataFrame
        """
        # Group by ticker and take top N errors by severity
        severity_order = {"error": 0, "warning": 1, "info": 2}

        sampled = (
            logs_df
            .with_columns([
                pl.col("severity").map_elements(
                    lambda x: severity_order.get(x, 3),
                    return_dtype=pl.Int32
                ).alias("_severity_rank")
            ])
            .sort(["ticker", "_severity_rank", "date"])
            .group_by("ticker")
            .head(max_per_ticker)
            .drop("_severity_rank")
        )

        print(f"  Sampled from {logs_df.height:,} to {sampled.height:,} errors")
        return sampled

    def _get_ticker_data(self, ticker: str, max_rows: int = 10_000) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Lazily load data for a specific ticker only when needed.

        Args:
            ticker: Ticker symbol
            max_rows: Maximum rows to load (for memory efficiency)

        Returns:
            Tuple of (original_df, cleaned_df)
        """
        if ticker not in self.original_dfs:
            return pl.DataFrame(), pl.DataFrame()

        # Collect only the specific ticker (lazy evaluation!)
        # Limit rows to prevent memory issues with very large datasets
        original = self.original_dfs[ticker].limit(max_rows).collect()
        cleaned = self.cleaned_dfs.get(ticker, self.original_dfs[ticker]).limit(max_rows).collect()

        return original, cleaned

    def load_false_positives(self):
        """Load false positive flags from storage."""
        if self.false_positive_store_path.exists():
            with open(self.false_positive_store_path, 'r') as f:
                fp_data = json.load(f)
                # Update normalized logs with false positive flags
                # (In production, would merge with database)

    def save_false_positive(self, ticker: str, date: str, error_type: str, is_fp: bool):
        """Save a false positive flag."""
        if self.false_positive_store_path.exists():
            with open(self.false_positive_store_path, 'r') as f:
                fp_data = json.load(f)
        else:
            fp_data = []

        fp_data.append({
            "ticker": ticker,
            "date": date,
            "error_type": error_type,
            "is_false_positive": is_fp,
            "flagged_at": datetime.now().isoformat()
        })

        with open(self.false_positive_store_path, 'w') as f:
            json.dump(fp_data, f, indent=2)

    def _build_layout(self):
        """Build the dashboard layout."""

        # Get unique values for filters
        tickers = sorted(self.normalized_logs["ticker"].unique().to_list()) if self.normalized_logs.height > 0 else []
        categories = sorted(self.normalized_logs["error_category"].unique().to_list()) if self.normalized_logs.height > 0 else []

        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Financial Data Cleaning Audit Dashboard",
                           className="text-center mb-4 mt-4"),
                    html.Hr()
                ])
            ]),

            # Main Content
            dbc.Row([
                # Sidebar
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(html.H4("Filters")),
                        dbc.CardBody([
                            # Ticker Dropdown
                            html.Label("Ticker", className="fw-bold"),
                            dcc.Dropdown(
                                id="ticker-dropdown",
                                options=[{"label": t, "value": t} for t in tickers],
                                value=tickers[0] if tickers else None,
                                placeholder="Select a ticker..."
                            ),
                            html.Br(),

                            # Error Category Dropdown
                            html.Label("Error Category", className="fw-bold"),
                            dcc.Dropdown(
                                id="category-dropdown",
                                options=[{"label": c, "value": c} for c in categories],
                                placeholder="All categories..."
                            ),
                            html.Br(),

                            # Specific Error Dropdown (Dynamic)
                            html.Label("Specific Error", className="fw-bold"),
                            dcc.Dropdown(
                                id="error-dropdown",
                                placeholder="Select error type..."
                            ),
                            html.Br(),

                            # Financial Equivalencies Toggle
                            html.Label("Financial Equivalencies Filter", className="fw-bold"),
                            dcc.RadioItems(
                                id="financial-filter-toggle",
                                options=[
                                    {"label": "All", "value": "all"},
                                    {"label": "Hard Errors Only", "value": "hard"},
                                    {"label": "Soft Warnings Only", "value": "soft"}
                                ],
                                value="all",
                                inline=True
                            ),
                            html.Br(),

                            # Summary Statistics
                            html.Hr(),
                            html.H5("Summary", className="fw-bold"),
                            html.Div(id="summary-stats")
                        ])
                    ])
                ], width=3),

                # Main Panel
                dbc.Col([
                    # Log Inspector
                    dbc.Card([
                        dbc.CardHeader(html.H4("Log Inspector")),
                        dbc.CardBody([
                            dag.AgGrid(
                                id="log-table",
                                columnDefs=[
                                    {"field": "ticker", "headerName": "Ticker", "width": 80},
                                    {"field": "date", "headerName": "Date", "width": 110},
                                    {"field": "error_type", "headerName": "Error Type", "width": 180},
                                    {"field": "column", "headerName": "Column(s)", "width": 200},
                                    {"field": "original_value", "headerName": "Original", "width": 120,
                                     "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                    {"field": "corrected_value", "headerName": "Corrected", "width": 120,
                                     "valueFormatter": {"function": "d3.format(',.2f')(params.value)"}},
                                    {"field": "message", "headerName": "Message", "flex": 1},
                                    {"field": "severity", "headerName": "Severity", "width": 100},
                                    {"field": "false_positive", "headerName": "False Positive", "width": 130}
                                ],
                                rowData=[],
                                defaultColDef={"sortable": True, "filter": True, "resizable": True},
                                dashGridOptions={
                                    "rowSelection": "single",
                                    "animateRows": False,
                                    "pagination": True,
                                    "paginationPageSize": 50,  # Limit rows per page for performance
                                    "cacheBlockSize": 50,  # Load 50 rows at a time
                                    "maxBlocksInCache": 10  # Keep max 500 rows in cache
                                },
                                style={"height": "400px"}
                            ),
                            html.Br(),
                            dbc.Button("Flag as False Positive", id="flag-fp-btn", color="warning", className="me-2"),
                            dbc.Button("Clear False Positive", id="clear-fp-btn", color="secondary"),
                            html.Div(id="flag-status", className="mt-2")
                        ])
                    ], className="mb-3"),

                    # Time Series Comparator
                    dbc.Card([
                        dbc.CardHeader(html.H4("Time Series Comparator")),
                        dbc.CardBody([
                            dcc.Graph(id="timeseries-chart", style={"height": "500px"})
                        ])
                    ])
                ], width=9)
            ]),

            # Hidden div to store selected row data
            html.Div(id="selected-row-data", style={"display": "none"}),

        ], fluid=True)

    def _register_callbacks(self):
        """Register all dashboard callbacks."""

        # Update error dropdown based on category selection
        @self.app.callback(
            Output("error-dropdown", "options"),
            Output("error-dropdown", "value"),
            Input("category-dropdown", "value"),
            Input("ticker-dropdown", "value")
        )
        def update_error_dropdown(category, ticker):
            if self.normalized_logs.height == 0:
                return [], None

            filtered = self.normalized_logs
            if ticker:
                filtered = filtered.filter(pl.col("ticker") == ticker)
            if category:
                filtered = filtered.filter(pl.col("error_category") == category)

            error_types = sorted(filtered["error_type"].unique().to_list())
            options = [{"label": e, "value": e} for e in error_types]

            return options, None

        # Update log table
        @self.app.callback(
            Output("log-table", "rowData"),
            Output("summary-stats", "children"),
            Input("ticker-dropdown", "value"),
            Input("category-dropdown", "value"),
            Input("error-dropdown", "value"),
            Input("financial-filter-toggle", "value")
        )
        def update_log_table(ticker, category, error_type, financial_filter):
            if self.normalized_logs.height == 0:
                return [], html.P("No data available")

            filtered = self.normalized_logs

            if ticker:
                filtered = filtered.filter(pl.col("ticker") == ticker)
            if category:
                filtered = filtered.filter(pl.col("error_category") == category)
            if error_type:
                filtered = filtered.filter(pl.col("error_type") == error_type)

            # Apply financial filter
            if financial_filter == "hard":
                filtered = filtered.filter(
                    pl.col("error_category") == "Financial Equivalencies (Hard)"
                )
            elif financial_filter == "soft":
                filtered = filtered.filter(
                    pl.col("error_category") == "Financial Equivalencies (Soft)"
                )

            # Limit rows for performance (take most recent/severe)
            MAX_DISPLAY_ROWS = 5000
            if filtered.height > MAX_DISPLAY_ROWS:
                # Keep most severe errors
                severity_order = {"error": 0, "warning": 1, "info": 2}
                filtered = (
                    filtered
                    .with_columns([
                        pl.col("severity").map_elements(
                            lambda x: severity_order.get(x, 3),
                            return_dtype=pl.Int32
                        ).alias("_sev_rank")
                    ])
                    .sort(["_sev_rank", "date"])
                    .head(MAX_DISPLAY_ROWS)
                    .drop("_sev_rank")
                )

            # Convert to dict for AG Grid
            row_data = filtered.to_dicts()

            # Summary stats
            total_errors = filtered.height
            unique_dates = filtered["date"].n_unique()
            severity_counts = filtered.group_by("severity").count().to_dicts()

            summary = [
                html.P(f"Total Errors: {total_errors}", className="mb-1"),
                html.P(f"Unique Dates: {unique_dates}", className="mb-1"),
                html.P("By Severity:", className="mb-1 fw-bold"),
            ]

            for s in severity_counts:
                summary.append(
                    html.P(f"  {s['severity']}: {s['count']}", className="mb-1 ms-3")
                )

            return row_data, summary

        # Update time series chart
        @self.app.callback(
            Output("timeseries-chart", "figure"),
            Input("ticker-dropdown", "value"),
            Input("log-table", "selectedRows"),
            Input("category-dropdown", "value"),
            State("error-dropdown", "value")
        )
        def update_timeseries(ticker, selected_rows, category, error_type):
            if not ticker or ticker not in self.original_dfs:
                return go.Figure().add_annotation(
                    text="Select a ticker to view time series",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )

            # Lazy load only when needed - saves memory!
            original_df, cleaned_df = self._get_ticker_data(ticker, max_rows=10_000)

            # Determine which column to plot
            column_to_plot = None
            selected_date = None
            error_dates = []

            if selected_rows and len(selected_rows) > 0:
                row = selected_rows[0]
                column_to_plot = row.get("column", "").split(",")[0].strip()
                selected_date = row.get("date")

            # If no specific column selected, try to infer from category
            if not column_to_plot and category:
                if "OHLC" in category:
                    column_to_plot = "m_close"
                elif "Negative Market" in category:
                    column_to_plot = "m_close"
                elif "Negative Fundamental" in category:
                    column_to_plot = "fbs_assets"
                elif "Financial Equivalencies" in category:
                    # Special handling for accounting equation (loads data internally)
                    return self._create_accounting_equation_chart(ticker)
                else:
                    column_to_plot = "m_close"

            if not column_to_plot:
                column_to_plot = "m_close"

            # Check if column exists
            if column_to_plot not in original_df.columns:
                # Try to find a similar column
                possible_columns = [c for c in original_df.columns if column_to_plot.split("_")[-1] in c]
                if possible_columns:
                    column_to_plot = possible_columns[0]
                else:
                    return go.Figure().add_annotation(
                        text=f"Column '{column_to_plot}' not found in data",
                        xref="paper", yref="paper",
                        x=0.5, y=0.5, showarrow=False
                    )

            # Get error dates for this column
            if self.normalized_logs.height > 0:
                error_df = self.normalized_logs.filter(
                    (pl.col("ticker") == ticker) &
                    (pl.col("column").str.contains(column_to_plot.split("_")[-1]))
                )
                error_dates = error_df["date"].to_list()

            # Create figure
            fig = go.Figure()

            # Add original data line
            fig.add_trace(go.Scatter(
                x=original_df["m_date"],
                y=original_df[column_to_plot],
                mode="lines",
                name="Original",
                line=dict(color="red", dash="dash", width=2)
            ))

            # Add cleaned data line
            fig.add_trace(go.Scatter(
                x=cleaned_df["m_date"],
                y=cleaned_df[column_to_plot],
                mode="lines",
                name="Cleaned",
                line=dict(color="green", width=2)
            ))

            # Add error markers
            if error_dates:
                error_df_filtered = original_df.filter(
                    pl.col("m_date").cast(pl.Utf8).is_in(error_dates)
                )

                if error_df_filtered.height > 0:
                    fig.add_trace(go.Scatter(
                        x=error_df_filtered["m_date"],
                        y=error_df_filtered[column_to_plot],
                        mode="markers",
                        name="Errors",
                        marker=dict(
                            symbol="x",
                            size=12,
                            color="orange",
                            line=dict(width=2)
                        )
                    ))

            # Update layout
            fig.update_layout(
                title=f"{ticker} - {column_to_plot}",
                xaxis_title="Date",
                yaxis_title="Value",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Zoom to selected date if available
            if selected_date:
                try:
                    date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
                    start_zoom = date_obj - timedelta(days=30)
                    end_zoom = date_obj + timedelta(days=30)

                    fig.update_xaxes(range=[start_zoom, end_zoom])
                except:
                    pass

            return fig

        # Flag false positive
        @self.app.callback(
            Output("flag-status", "children"),
            Input("flag-fp-btn", "n_clicks"),
            Input("clear-fp-btn", "n_clicks"),
            State("log-table", "selectedRows"),
            prevent_initial_call=True
        )
        def handle_false_positive(flag_clicks, clear_clicks, selected_rows):
            if not selected_rows or len(selected_rows) == 0:
                return dbc.Alert("Please select a row first", color="warning", duration=3000)

            row = selected_rows[0]
            ticker = row["ticker"]
            date = row["date"]
            error_type = row["error_type"]

            triggered_id = ctx.triggered_id

            if triggered_id == "flag-fp-btn":
                self.save_false_positive(ticker, date, error_type, True)
                return dbc.Alert("Flagged as false positive", color="success", duration=3000)
            elif triggered_id == "clear-fp-btn":
                self.save_false_positive(ticker, date, error_type, False)
                return dbc.Alert("False positive flag cleared", color="info", duration=3000)

            return ""

    def _create_accounting_equation_chart(self, ticker: str) -> go.Figure:
        """
        Create special chart for accounting equation: Assets vs (Liabilities + Equity + NCI)
        """
        # Lazy load data only when needed
        original_df, cleaned_df = self._get_ticker_data(ticker, max_rows=10_000)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Accounting Equation Components", "Difference (Assets - Claims)"),
            vertical_spacing=0.15,
            row_heights=[0.7, 0.3]
        )

        # Check if required columns exist
        required_cols = ["fbs_assets", "fbs_liabilities", "fbs_stockholder_equity"]
        if not all(col in original_df.columns for col in required_cols):
            return go.Figure().add_annotation(
                text="Required accounting columns not found in data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )

        # Calculate total claims
        original_df = original_df.with_columns(
            (pl.col("fbs_liabilities") +
             pl.col("fbs_stockholder_equity") +
             pl.col("fbs_noncontrolling_interest").fill_null(0)).alias("total_claims")
        )

        cleaned_df = cleaned_df.with_columns(
            (pl.col("fbs_liabilities") +
             pl.col("fbs_stockholder_equity") +
             pl.col("fbs_noncontrolling_interest").fill_null(0)).alias("total_claims")
        )

        # Plot 1: Assets vs Total Claims
        fig.add_trace(
            go.Scatter(
                x=original_df["m_date"],
                y=original_df["fbs_assets"],
                mode="lines",
                name="Assets (Original)",
                line=dict(color="blue", dash="dash")
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=original_df["m_date"],
                y=original_df["total_claims"],
                mode="lines",
                name="Liabilities + Equity + NCI (Original)",
                line=dict(color="red", dash="dash")
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=cleaned_df["m_date"],
                y=cleaned_df["total_claims"],
                mode="lines",
                name="Liabilities + Equity + NCI (Cleaned)",
                line=dict(color="green")
            ),
            row=1, col=1
        )

        # Plot 2: Difference
        original_diff = original_df["fbs_assets"] - original_df["total_claims"]
        cleaned_diff = cleaned_df["fbs_assets"] - cleaned_df["total_claims"]

        fig.add_trace(
            go.Scatter(
                x=original_df["m_date"],
                y=original_diff,
                mode="lines",
                name="Difference (Original)",
                line=dict(color="orange", dash="dash")
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=cleaned_df["m_date"],
                y=cleaned_diff,
                mode="lines",
                name="Difference (Cleaned)",
                line=dict(color="purple")
            ),
            row=2, col=1
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(
            title=f"{ticker} - Accounting Equation Analysis",
            height=700,
            hovermode="x unified",
            showlegend=True
        )

        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Difference ($)", row=2, col=1)

        return fig

    def run(self, debug=True, port=8050):
        """Run the dashboard application."""
        self.app.run(debug=debug, port=port)
