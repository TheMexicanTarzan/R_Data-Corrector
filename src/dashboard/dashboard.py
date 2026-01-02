"""
Data Corrector Dashboard

A robust, interactive Dash application to visualize the results of the financial
data cleaning pipeline. Serves as an audit tool, allowing users to compare original
vs. cleaned data and inspect specific error logs.
"""

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional, Union

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import polars
from dash import Input, Output, State, dcc, html

# ============================================================================
# CONSTANTS
# ============================================================================

ERROR_CATEGORIES = {
    "Date Sorting": "unsorted_dates_logs",
    "Negative Fundamentals": "negative_fundamentals_logs",
    "Negative Market Data": "negative_market_logs",
    "Zero Wipeout": "zero_wipeout_logs",
    "Market Cap Scale": "shares_outstanding_logs",
    "OHLC Integrity": "ohlc_logs",
    "Accounting Mismatch": "financial_unequivalencies_logs",
    "Split Consistency": "split_inconsistencies_logs",
}

# Path for storing false positive flags
FALSE_POSITIVES_FILE = Path(__file__).parent / "false_positives.json"

# Maximum rows to display in the grid to prevent memory issues
MAX_DISPLAY_ROWS = 500


# ============================================================================
# LOG NORMALIZER FUNCTIONS
# ============================================================================

def serialize_date(value: Any) -> str:
    """Convert date objects to string for display."""
    if isinstance(value, (date, datetime)):
        return value.strftime("%Y-%m-%d")
    return str(value) if value is not None else ""


def _preprocess_parallel_logs(logs: dict) -> dict:
    """
    Preprocess logs from parallel_process_tickers to flatten/merge structures.

    parallel_process_tickers returns logs as list[audit_log_per_ticker], so we need to:
    - Flatten list[list[dict]] -> list[dict] for most log types
    - Merge list[dict[str, list]] -> dict[str, list] for negative_fundamentals
    - Merge list[dict] with hard/soft keys -> single dict for financial_unequivalencies
    """
    processed = {}

    for log_key, log_data in logs.items():
        if not log_data:
            processed[log_key] = []
            continue

        # Check if this is a list of per-ticker results
        if not isinstance(log_data, list):
            processed[log_key] = log_data
            continue

        if log_key == "negative_fundamentals_logs":
            # Merge list[dict[str, list[dict]]] -> dict[str, list[dict]]
            merged = {}
            for ticker_log in log_data:
                if isinstance(ticker_log, dict):
                    for col_name, entries in ticker_log.items():
                        if col_name not in merged:
                            merged[col_name] = []
                        if isinstance(entries, list):
                            merged[col_name].extend(entries)
            processed[log_key] = merged

        elif log_key == "financial_unequivalencies_logs":
            # Merge list[dict] -> single dict with combined hard/soft lists
            merged = {"hard_filter_errors": [], "soft_filter_warnings": []}
            for ticker_log in log_data:
                if isinstance(ticker_log, dict):
                    if "hard_filter_errors" in ticker_log:
                        merged["hard_filter_errors"].extend(ticker_log["hard_filter_errors"])
                    if "soft_filter_warnings" in ticker_log:
                        merged["soft_filter_warnings"].extend(ticker_log["soft_filter_warnings"])
            processed[log_key] = merged

        else:
            # Flatten list[list[dict]] -> list[dict]
            flattened = []
            for ticker_log in log_data:
                if isinstance(ticker_log, list):
                    flattened.extend(ticker_log)
                elif isinstance(ticker_log, dict):
                    # Single dict entry
                    flattened.append(ticker_log)
            processed[log_key] = flattened

    return processed


def normalize_logs(logs: dict) -> polars.LazyFrame:
    """
    Parse all log structures into a unified Polars LazyFrame.

    Handles different log structures:
    - Flat lists (most logs)
    - Nested dicts by column (negative_fundamentals_logs)
    - Dict with hard_filter_errors/soft_filter_warnings (financial_unequivalencies_logs)

    Returns:
        LazyFrame with columns: ticker, date, error_category, error_type,
        column_involved, original_value, corrected_value, message, raw_log
    """
    # Preprocess logs from parallel processing
    logs = _preprocess_parallel_logs(logs)

    normalized_rows = []

    # Process each log category
    for category_name, log_key in ERROR_CATEGORIES.items():
        log_data = logs.get(log_key, [])

        if not log_data:
            continue

        if log_key == "negative_fundamentals_logs":
            # Structure: dict[column_name, list[dict]]
            normalized_rows.extend(
                _normalize_negative_fundamentals(log_data, category_name)
            )
        elif log_key == "financial_unequivalencies_logs":
            # Structure: dict with 'hard_filter_errors' and 'soft_filter_warnings'
            normalized_rows.extend(
                _normalize_financial_equivalencies(log_data, category_name)
            )
        elif isinstance(log_data, list):
            # Standard flat list structure
            normalized_rows.extend(
                _normalize_flat_logs(log_data, category_name, log_key)
            )

    if not normalized_rows:
        # Return empty LazyFrame with correct schema
        return polars.DataFrame({
            "ticker": [],
            "date": [],
            "error_category": [],
            "error_type": [],
            "column_involved": [],
            "original_value": [],
            "corrected_value": [],
            "message": [],
            "filter_type": [],
            "raw_log": [],
        }).lazy()

    return polars.DataFrame(normalized_rows).lazy()


def _normalize_negative_fundamentals(log_data: dict, category_name: str) -> list[dict]:
    """Normalize negative fundamentals logs (nested by column)."""
    rows = []

    if not isinstance(log_data, dict):
        return rows

    for column_name, entries in log_data.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            # Get the date - could be 'm_date' or 'date'
            date_val = entry.get("m_date") or entry.get("date")
            original_value = entry.get(column_name)

            rows.append({
                "ticker": entry.get("ticker", ""),
                "date": serialize_date(date_val),
                "error_category": category_name,
                "error_type": "negative_value",
                "column_involved": column_name,
                "original_value": str(original_value) if original_value is not None else "",
                "corrected_value": "Forward filled",
                "message": f"Negative value {original_value} in {column_name}",
                "filter_type": "correction",
                "raw_log": json.dumps(entry, default=str),
            })

    return rows


def _normalize_financial_equivalencies(log_data: dict, category_name: str) -> list[dict]:
    """Normalize financial equivalencies logs (hard/soft filter structure)."""
    rows = []

    if not isinstance(log_data, dict):
        return rows

    # Process hard filter errors (corrections)
    hard_errors = log_data.get("hard_filter_errors", [])
    for entry in hard_errors:
        date_val = entry.get("date")
        error_type = entry.get("error_type", "unknown")

        # Build message based on error type
        if error_type == "assets_mismatch":
            message = (
                f"Assets mismatch: Total={entry.get('total')}, "
                f"Current+Noncurrent={entry.get('component_sum')}, "
                f"Diff={entry.get('difference')}"
            )
            column_involved = "fbs_assets/fbs_current_assets/fbs_noncurrent_assets"
        elif error_type == "liabilities_mismatch":
            message = (
                f"Liabilities mismatch: Total={entry.get('total')}, "
                f"Current+Noncurrent={entry.get('component_sum')}, "
                f"Diff={entry.get('difference')}"
            )
            column_involved = "fbs_liabilities/fbs_current_liabilities/fbs_noncurrent_liabilities"
        else:
            message = f"Error: {error_type}"
            column_involved = ""

        original_value = f"Current={entry.get('current')}, Noncurrent={entry.get('noncurrent')}"
        corrected_value = (
            f"Current={entry.get('corrected_current')}, "
            f"Noncurrent={entry.get('corrected_noncurrent')}"
        )

        rows.append({
            "ticker": entry.get("ticker", ""),
            "date": serialize_date(date_val),
            "error_category": category_name,
            "error_type": error_type,
            "column_involved": column_involved,
            "original_value": original_value,
            "corrected_value": corrected_value,
            "message": message,
            "filter_type": "hard_error",
            "raw_log": json.dumps(entry, default=str),
        })

    # Process soft filter warnings (flags only)
    soft_warnings = log_data.get("soft_filter_warnings", [])
    for entry in soft_warnings:
        date_val = entry.get("date")
        error_type = entry.get("error_type", "unknown")

        if error_type == "equity_mismatch":
            components = entry.get("components", {})
            message = (
                f"Equity mismatch: Total={entry.get('total')}, "
                f"Sum={entry.get('component_sum')}, Diff={entry.get('difference')}"
            )
            column_involved = "fbs_stockholder_equity"
            original_value = str(components)
        elif error_type == "cash_mismatch":
            message = (
                f"Cash mismatch: Period End Cash={entry.get('fcf_period_end_cash')}, "
                f"Cash & Equiv={entry.get('fbs_cash_and_cash_equivalents')}, "
                f"Diff={entry.get('difference')}"
            )
            column_involved = "fcf_period_end_cash/fbs_cash_and_cash_equivalents"
            original_value = (
                f"Period End={entry.get('fcf_period_end_cash')}, "
                f"Cash&Equiv={entry.get('fbs_cash_and_cash_equivalents')}"
            )
        elif error_type == "accounting_equation_mismatch":
            components = entry.get("components", {})
            message = (
                f"A!=L+E+NCI: Assets={entry.get('assets (total)')}, "
                f"Claims={entry.get('claims_sum')}, Diff={entry.get('difference')}"
            )
            column_involved = "fbs_assets/fbs_liabilities/fbs_stockholder_equity/fbs_noncontrolling_interest"
            original_value = str(components)
        else:
            message = f"Warning: {error_type}"
            column_involved = ""
            original_value = ""

        rows.append({
            "ticker": entry.get("ticker", ""),
            "date": serialize_date(date_val),
            "error_category": category_name,
            "error_type": error_type,
            "column_involved": column_involved,
            "original_value": original_value,
            "corrected_value": "Not applicable",
            "message": message,
            "filter_type": "soft_warning",
            "raw_log": json.dumps(entry, default=str),
        })

    return rows


def _normalize_flat_logs(log_data: list, category_name: str, log_key: str) -> list[dict]:
    """Normalize flat list log structures."""
    rows = []

    for entry in log_data:
        if not isinstance(entry, dict):
            continue

        # Extract common fields
        ticker = entry.get("ticker", "")
        date_val = entry.get("date") or entry.get("m_date")
        error_type = entry.get("error_type", "")

        # Determine column involved and values based on log type
        if log_key == "unsorted_dates_logs":
            rows.append(_normalize_unsorted_dates_entry(entry, category_name, ticker, date_val))
        elif log_key == "negative_market_logs":
            rows.append(_normalize_negative_market_entry(entry, category_name, ticker, date_val))
        elif log_key == "zero_wipeout_logs":
            rows.append(_normalize_zero_wipeout_entry(entry, category_name, ticker, date_val))
        elif log_key == "shares_outstanding_logs":
            rows.append(_normalize_shares_outstanding_entry(entry, category_name, ticker, date_val))
        elif log_key == "ohlc_logs":
            rows.append(_normalize_ohlc_entry(entry, category_name, ticker, date_val))
        elif log_key == "split_inconsistencies_logs":
            rows.append(_normalize_split_consistency_entry(entry, category_name, ticker, date_val))
        else:
            # Generic fallback
            rows.append({
                "ticker": ticker,
                "date": serialize_date(date_val),
                "error_category": category_name,
                "error_type": error_type,
                "column_involved": "",
                "original_value": "",
                "corrected_value": "Not applicable",
                "message": entry.get("message", str(entry)),
                "filter_type": "correction",
                "raw_log": json.dumps(entry, default=str),
            })

    return rows


def _normalize_unsorted_dates_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize unsorted dates log entry."""
    error_type = entry.get("error_type", "")

    if error_type == "order_mismatch":
        message = (
            f"Row moved from position {entry.get('original_position')} "
            f"to {entry.get('sorted_position')}"
        )
        original_value = str(entry.get("original_position"))
        corrected_value = str(entry.get("sorted_position"))
        column_involved = "m_date/f_filing_date"
    elif error_type == "duplicates_removed":
        message = (
            f"Removed {entry.get('duplicates_removed')} duplicates "
            f"using strategy: {entry.get('strategy')}"
        )
        original_value = str(entry.get("duplicates_removed"))
        corrected_value = "Deduplicated"
        column_involved = "m_date"
        date_val = None  # No specific date for this type
    elif error_type == "no_date_columns":
        message = entry.get("message", "No date columns found")
        original_value = ""
        corrected_value = "Not applicable"
        column_involved = ""
        date_val = None
    else:
        message = entry.get("message", str(entry))
        original_value = ""
        corrected_value = "Not applicable"
        column_involved = ""

    return {
        "ticker": ticker,
        "date": serialize_date(date_val) if date_val else "",
        "error_category": category_name,
        "error_type": error_type,
        "column_involved": column_involved,
        "original_value": original_value,
        "corrected_value": corrected_value,
        "message": message,
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def _normalize_negative_market_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize negative market data log entry."""
    column = entry.get("column", "")
    original_value = entry.get("original_value")
    corrected_value = entry.get("corrected_value")
    method = entry.get("method", "unknown")

    message = f"Negative value corrected using {method}"

    return {
        "ticker": ticker,
        "date": serialize_date(date_val),
        "error_category": category_name,
        "error_type": "negative_market_value",
        "column_involved": column,
        "original_value": str(original_value) if original_value is not None else "",
        "corrected_value": str(corrected_value) if corrected_value is not None else "Skipped",
        "message": message,
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def _normalize_zero_wipeout_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize zero wipeout log entry."""
    # Find which columns were zero
    zero_columns = []
    for key, value in entry.items():
        if key not in ["ticker", "m_date", "date", "m_volume"] and value == 0:
            zero_columns.append(key)

    column_involved = ", ".join(zero_columns) if zero_columns else "shares columns"
    volume = entry.get("m_volume", 0)

    return {
        "ticker": ticker,
        "date": serialize_date(date_val),
        "error_category": category_name,
        "error_type": "zero_with_volume",
        "column_involved": column_involved,
        "original_value": f"0 (volume={volume})",
        "corrected_value": "Forward filled",
        "message": f"Zero shares with positive volume ({volume})",
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def _normalize_shares_outstanding_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize shares outstanding (10x jump) log entry."""
    error_type = entry.get("error_type", "scale_10x_jump")

    # Find the columns involved (exclude metadata)
    columns = [k for k in entry.keys() if k not in ["ticker", "date", "m_date", "error_type"]]
    column_involved = ", ".join(columns) if columns else "shares/market_cap"

    # Build values string
    values = {k: entry.get(k) for k in columns if entry.get(k) is not None}

    return {
        "ticker": ticker,
        "date": serialize_date(date_val),
        "error_category": category_name,
        "error_type": error_type,
        "column_involved": column_involved,
        "original_value": str(values) if values else "",
        "corrected_value": "Forward filled",
        "message": "10x scale jump detected and corrected",
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def _normalize_ohlc_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize OHLC integrity log entry."""
    error_type = entry.get("error_type", "")
    column_group = entry.get("column_group", "")
    message = entry.get("message", "")

    if error_type == "high_not_maximum":
        old_val = entry.get("old_high")
        new_val = entry.get("new_high")
        column_involved = f"m_high ({column_group})"
    elif error_type == "low_not_minimum":
        old_val = entry.get("old_low")
        new_val = entry.get("new_low")
        column_involved = f"m_low ({column_group})"
    elif error_type == "vwap_outside_range":
        old_val = entry.get("old_vwap")
        new_val = entry.get("new_vwap")
        column_involved = f"m_vwap ({column_group})"
    else:
        old_val = ""
        new_val = ""
        column_involved = column_group

    return {
        "ticker": ticker,
        "date": serialize_date(date_val),
        "error_category": category_name,
        "error_type": error_type,
        "column_involved": column_involved,
        "original_value": str(old_val) if old_val is not None else "",
        "corrected_value": str(new_val) if new_val is not None else "",
        "message": message,
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def _normalize_split_consistency_entry(entry: dict, category_name: str, ticker: str, date_val: Any) -> dict:
    """Normalize split consistency log entry."""
    error_type = entry.get("error_type", "")

    if error_type in ["skipped_validation", "skipped_pair"]:
        reason = entry.get("reason", "")
        missing = entry.get("missing", [])
        message = f"Skipped: {reason}. Missing: {missing}"
        return {
            "ticker": ticker,
            "date": "",
            "error_category": category_name,
            "error_type": error_type,
            "column_involved": ", ".join(missing) if missing else "",
            "original_value": "",
            "corrected_value": "Not applicable",
            "message": message,
            "filter_type": "info",
            "raw_log": json.dumps(entry, default=str),
        }

    raw_col = entry.get("raw_column", "")
    adj_col = entry.get("adjusted_column", "")
    original_adj = entry.get("original_adjusted_value")
    corrected_adj = entry.get("corrected_adjusted_value")
    k_expected = entry.get("k_expected")
    k_implied = entry.get("k_implied")

    message = (
        f"Split mismatch: K_expected={k_expected:.6f}, K_implied={k_implied:.6f}"
        if k_expected and k_implied else "Split mismatch detected"
    )

    return {
        "ticker": ticker,
        "date": serialize_date(date_val),
        "error_category": category_name,
        "error_type": error_type,
        "column_involved": f"{raw_col}/{adj_col}",
        "original_value": str(original_adj) if original_adj is not None else "",
        "corrected_value": str(corrected_adj) if corrected_adj is not None else "",
        "message": message,
        "filter_type": "correction",
        "raw_log": json.dumps(entry, default=str),
    }


def get_unique_tickers(logs: dict) -> list[str]:
    """Extract unique tickers from all logs."""
    # Preprocess logs from parallel processing first
    logs = _preprocess_parallel_logs(logs)

    tickers = set()

    for log_key in ERROR_CATEGORIES.values():
        log_data = logs.get(log_key, [])

        if log_key == "negative_fundamentals_logs" and isinstance(log_data, dict):
            for entries in log_data.values():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and "ticker" in entry:
                            tickers.add(entry["ticker"])
        elif log_key == "financial_unequivalencies_logs" and isinstance(log_data, dict):
            for key in ["hard_filter_errors", "soft_filter_warnings"]:
                for entry in log_data.get(key, []):
                    if isinstance(entry, dict) and "ticker" in entry:
                        tickers.add(entry["ticker"])
        elif isinstance(log_data, list):
            for entry in log_data:
                if isinstance(entry, dict) and "ticker" in entry:
                    tickers.add(entry["ticker"])

    return sorted(list(tickers))


# ============================================================================
# FALSE POSITIVES MANAGEMENT
# ============================================================================

def load_false_positives() -> dict:
    """Load false positives from JSON file."""
    if FALSE_POSITIVES_FILE.exists():
        try:
            with open(FALSE_POSITIVES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_false_positives(false_positives: dict) -> None:
    """Save false positives to JSON file."""
    FALSE_POSITIVES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FALSE_POSITIVES_FILE, "w") as f:
        json.dump(false_positives, f, indent=2)


def get_false_positive_key(ticker: str, date_str: str, error_type: str, column: str) -> str:
    """Generate a unique key for a potential false positive."""
    return f"{ticker}|{date_str}|{error_type}|{column}"


# ============================================================================
# LAYOUT COMPONENTS
# ============================================================================

def create_sidebar() -> dbc.Col:
    """Create the sidebar with filter controls."""
    return dbc.Col(
        [
            html.H4("Filters", className="mb-3"),

            # Ticker Dropdown with search
            html.Label("Ticker", className="fw-bold"),
            dcc.Dropdown(
                id="ticker-dropdown",
                placeholder="Search and select ticker...",
                searchable=True,
                clearable=True,
                className="mb-3",
            ),

            # Error Category Dropdown
            html.Label("Error Category", className="fw-bold"),
            dcc.Dropdown(
                id="error-category-dropdown",
                options=[{"label": k, "value": k} for k in ERROR_CATEGORIES.keys()],
                placeholder="Select error category...",
                searchable=True,
                clearable=True,
                className="mb-3",
            ),

            # Hard/Soft Filter Toggle (only for Accounting Mismatch)
            html.Div(
                id="filter-type-container",
                children=[
                    html.Label("Filter Type", className="fw-bold"),
                    dbc.RadioItems(
                        id="filter-type-toggle",
                        options=[
                            {"label": "Hard Errors (Corrections)", "value": "hard_error"},
                            {"label": "Soft Warnings (Flags)", "value": "soft_warning"},
                            {"label": "All", "value": "all"},
                        ],
                        value="all",
                        inline=False,
                        className="mb-3",
                    ),
                ],
                style={"display": "none"},
            ),

            # Specific Error Filter (Dynamic)
            html.Label("Specific Error", className="fw-bold"),
            dcc.Dropdown(
                id="specific-error-dropdown",
                placeholder="Filter by date/column...",
                searchable=True,
                clearable=True,
                multi=True,
                className="mb-3",
            ),

            html.Hr(),

            # Statistics
            html.Div(id="filter-stats", className="small text-muted"),
        ],
        width=3,
        className="bg-light p-3",
        style={"height": "100vh", "overflowY": "auto"},
    )


def create_log_inspector() -> html.Div:
    """Create the Log Inspector component using AG Grid."""
    column_defs = [
        {
            "field": "ticker",
            "headerName": "Ticker",
            "width": 80,
            "filter": True,
            "sortable": True,
        },
        {
            "field": "date",
            "headerName": "Date",
            "width": 110,
            "filter": True,
            "sortable": True,
        },
        {
            "field": "error_type",
            "headerName": "Error Type",
            "width": 150,
            "filter": True,
            "sortable": True,
        },
        {
            "field": "column_involved",
            "headerName": "Column(s)",
            "width": 200,
            "filter": True,
            "sortable": True,
        },
        {
            "field": "original_value",
            "headerName": "Original Value",
            "width": 150,
            "filter": True,
        },
        {
            "field": "corrected_value",
            "headerName": "Corrected Value",
            "width": 150,
            "filter": True,
        },
        {
            "field": "message",
            "headerName": "Message",
            "flex": 1,
            "minWidth": 200,
            "filter": True,
        },
        {
            "field": "is_false_positive",
            "headerName": "False Positive",
            "width": 120,
            "cellDataType": "boolean",
            "editable": False,  # Use button instead for toggling
            "cellStyle": {"textAlign": "center"},
        },
    ]

    return html.Div(
        [
            html.H5("Log Inspector", className="mb-2"),
            dag.AgGrid(
                id="log-grid",
                columnDefs=column_defs,
                rowData=[],
                defaultColDef={
                    "resizable": True,
                    "sortable": True,
                    "filter": True,
                },
                dashGridOptions={
                    "rowSelection": "single",
                    "animateRows": True,
                    "pagination": True,
                    "paginationPageSize": 20,
                    "domLayout": "autoHeight",
                },
                style={"height": "350px"},
                className="ag-theme-alpine",
            ),
            # False positive action button
            dbc.Button(
                "Toggle False Positive",
                id="toggle-false-positive-btn",
                color="warning",
                size="sm",
                className="mt-2",
                disabled=True,
            ),
            html.Div(id="false-positive-status", className="mt-2 small"),
        ]
    )


def create_error_visualization() -> html.Div:
    """Create the Error Visualization chart component."""
    return html.Div(
        [
            html.H5("Error Visualization", className="mb-2"),

            # Column selector for visualization
            html.Div(
                [
                    html.Label("Column to Plot:", className="me-2"),
                    dcc.Dropdown(
                        id="visualization-column-dropdown",
                        placeholder="Select column...",
                        searchable=True,
                        clearable=False,
                        className="d-inline-block",
                        style={"width": "300px"},
                    ),
                ],
                className="mb-2 d-flex align-items-center",
            ),

            # The chart
            dcc.Graph(
                id="error-chart",
                config={
                    "displayModeBar": True,
                    "scrollZoom": True,
                },
                style={"height": "400px"},
            ),

            # Legend
            html.Div(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "3px",
                            "backgroundColor": "red",
                            "marginRight": "5px",
                            "verticalAlign": "middle",
                            "borderTop": "2px dashed red",
                        }
                    ),
                    html.Span("Original Data  ", className="me-3"),
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "20px",
                            "height": "3px",
                            "backgroundColor": "green",
                            "marginRight": "5px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span("Cleaned Data  ", className="me-3"),
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "15px",
                            "height": "15px",
                            "backgroundColor": "rgba(255,0,0,0.2)",
                            "marginRight": "5px",
                            "verticalAlign": "middle",
                        }
                    ),
                    html.Span("Error Date"),
                ],
                className="small text-muted mt-2",
            ),
        ]
    )


def create_layout() -> dbc.Container:
    """Create the main dashboard layout."""
    return dbc.Container(
        [
            # Header
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.H2("Data Corrector", className="mb-0"),
                            html.P(
                                "Financial Data Cleaning Audit Dashboard",
                                className="text-muted mb-0",
                            ),
                        ],
                        width=12,
                    ),
                ],
                className="py-3 border-bottom mb-3",
            ),

            # Main content
            dbc.Row(
                [
                    # Sidebar
                    create_sidebar(),

                    # Main panel
                    dbc.Col(
                        [
                            # Log Inspector (Upper Section)
                            html.Div(create_log_inspector(), className="mb-4"),

                            # Error Visualization (Lower Section)
                            html.Div(create_error_visualization()),
                        ],
                        width=9,
                        className="p-3",
                    ),
                ],
                className="g-0",
            ),

            # Hidden stores for data
            dcc.Store(id="normalized-logs-store"),
            dcc.Store(id="original-dataframes-store"),
            dcc.Store(id="cleaned-dataframes-store"),
            dcc.Store(id="selected-row-store"),
            dcc.Store(id="false-positives-store"),
        ],
        fluid=True,
        className="px-0",
    )


# ============================================================================
# DASH APPLICATION FACTORY
# ============================================================================

def create_app(
    original_dataframes: dict[str, polars.DataFrame],
    cleaned_dataframes: dict[str, Union[polars.DataFrame, polars.LazyFrame]],
    logs: dict,
) -> dash.Dash:
    """
    Create and configure the Dash application.

    Args:
        original_dataframes: Dict of ticker -> original DataFrame
        cleaned_dataframes: Dict of ticker -> cleaned DataFrame/LazyFrame
        logs: Dict of log category -> log entries

    Returns:
        Configured Dash application
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.title = "Data Corrector"
    app.layout = create_layout()

    # Normalize logs once - keep as LazyFrame to avoid memory issues
    normalized_logs_lf = normalize_logs(logs)

    # Get unique tickers (this is lightweight - just extracts ticker names)
    unique_tickers = get_unique_tickers(logs)

    # Load false positives into a mutable container
    # Using a dict wrapper to allow mutation in callbacks
    false_positives_container = {"data": load_false_positives()}

    # Store total log count for stats (using count which is memory efficient)
    try:
        total_log_count = normalized_logs_lf.select(polars.len()).collect().item()
    except Exception:
        total_log_count = 0

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    @app.callback(
        [
            Output("ticker-dropdown", "options"),
            Output("false-positives-store", "data"),
        ],
        Input("normalized-logs-store", "data"),
    )
    def initialize_app(_):
        """Initialize app with ticker options and false positives."""
        return (
            [{"label": t, "value": t} for t in unique_tickers],
            false_positives_container["data"],
        )

    @app.callback(
        Output("filter-type-container", "style"),
        Input("error-category-dropdown", "value"),
    )
    def toggle_filter_type_visibility(category: Optional[str]):
        """Show/hide the filter type toggle based on category."""
        if category == "Accounting Mismatch":
            return {"display": "block"}
        return {"display": "none"}

    @app.callback(
        Output("specific-error-dropdown", "options"),
        [
            Input("ticker-dropdown", "value"),
            Input("error-category-dropdown", "value"),
            Input("filter-type-toggle", "value"),
        ],
    )
    def populate_specific_error_dropdown(
        ticker: Optional[str],
        category: Optional[str],
        filter_type: str,
    ):
        """Populate specific error dropdown based on current filters."""
        if total_log_count == 0:
            return []

        # Start with lazy filtering
        filtered_lf = normalized_logs_lf

        if ticker:
            filtered_lf = filtered_lf.filter(polars.col("ticker") == ticker)

        if category:
            filtered_lf = filtered_lf.filter(polars.col("error_category") == category)

        if category == "Accounting Mismatch" and filter_type != "all":
            filtered_lf = filtered_lf.filter(polars.col("filter_type") == filter_type)

        # Get unique values efficiently using group_by - limit to prevent memory issues
        try:
            unique_dates = (
                filtered_lf.select("date")
                .filter(polars.col("date").is_not_null() & (polars.col("date") != ""))
                .unique()
                .limit(100)
                .collect()["date"].to_list()
            )
            unique_columns = (
                filtered_lf.select("column_involved")
                .filter(polars.col("column_involved").is_not_null() & (polars.col("column_involved") != ""))
                .unique()
                .limit(50)
                .collect()["column_involved"].to_list()
            )
            unique_types = (
                filtered_lf.select("error_type")
                .filter(polars.col("error_type").is_not_null() & (polars.col("error_type") != ""))
                .unique()
                .limit(20)
                .collect()["error_type"].to_list()
            )
        except Exception:
            return []

        # Create options
        options = []
        for date_str in sorted(unique_dates)[:100]:
            options.append({"label": f"Date: {date_str}", "value": f"date:{date_str}"})
        for column in sorted(unique_columns)[:50]:
            options.append({"label": f"Column: {column}", "value": f"column:{column}"})
        for error_type in sorted(unique_types)[:20]:
            options.append({"label": f"Type: {error_type}", "value": f"type:{error_type}"})

        return options

    @app.callback(
        [
            Output("log-grid", "rowData"),
            Output("filter-stats", "children"),
        ],
        [
            Input("ticker-dropdown", "value"),
            Input("error-category-dropdown", "value"),
            Input("filter-type-toggle", "value"),
            Input("specific-error-dropdown", "value"),
            Input("false-positives-store", "data"),
        ],
    )
    def update_log_grid(
        ticker: Optional[str],
        category: Optional[str],
        filter_type: str,
        specific_filters: Optional[list],
        fp_store_data: Optional[dict],
    ):
        """Update the log grid based on filters."""
        if total_log_count == 0:
            return [], "No logs available"

        # Require at least one filter to prevent loading too much data
        if not ticker and not category and not specific_filters:
            return [], f"Please select a ticker or category to view logs ({total_log_count:,} total entries)"

        # Start with lazy filtering
        filtered_lf = normalized_logs_lf

        if ticker:
            filtered_lf = filtered_lf.filter(polars.col("ticker") == ticker)

        if category:
            filtered_lf = filtered_lf.filter(polars.col("error_category") == category)

        if category == "Accounting Mismatch" and filter_type != "all":
            filtered_lf = filtered_lf.filter(polars.col("filter_type") == filter_type)

        # Apply specific filters
        if specific_filters:
            for filter_val in specific_filters:
                if filter_val.startswith("date:"):
                    date_val = filter_val[5:]
                    filtered_lf = filtered_lf.filter(polars.col("date") == date_val)
                elif filter_val.startswith("column:"):
                    col_val = filter_val[7:]
                    filtered_lf = filtered_lf.filter(polars.col("column_involved") == col_val)
                elif filter_val.startswith("type:"):
                    type_val = filter_val[5:]
                    filtered_lf = filtered_lf.filter(polars.col("error_type") == type_val)

        # Get count before limiting (efficiently)
        try:
            match_count = filtered_lf.select(polars.len()).collect().item()
        except Exception:
            match_count = 0

        if match_count == 0:
            return [], "No matching logs found"

        # Collect with limit to prevent memory issues
        try:
            filtered_df = filtered_lf.limit(MAX_DISPLAY_ROWS).collect()
        except Exception as e:
            return [], f"Error loading logs: {str(e)[:100]}"

        # Get current false positives from store
        current_fp = fp_store_data or false_positives_container["data"]

        # Add false positive flags
        rows = filtered_df.to_dicts()
        for row in rows:
            fp_key = get_false_positive_key(
                row.get("ticker", ""),
                row.get("date", ""),
                row.get("error_type", ""),
                row.get("column_involved", ""),
            )
            row["is_false_positive"] = fp_key in current_fp

        # Build stats message
        if match_count > MAX_DISPLAY_ROWS:
            stats = f"Showing {len(rows)} of {match_count:,} log entries"
        else:
            stats = f"Showing {len(rows)} log entries"
        if ticker:
            stats += f" for {ticker}"
        if category:
            stats += f" ({category})"

        return rows, stats

    @app.callback(
        Output("visualization-column-dropdown", "options"),
        [
            Input("ticker-dropdown", "value"),
            Input("log-grid", "selectedRows"),
        ],
    )
    def populate_visualization_column_dropdown(
        ticker: Optional[str],
        selected_rows: Optional[list],
    ):
        """Populate the column dropdown for visualization."""
        if not ticker:
            return []

        # Get columns from cleaned dataframes
        if ticker in cleaned_dataframes:
            df = cleaned_dataframes[ticker]
            if isinstance(df, polars.LazyFrame):
                cols = df.collect_schema().names()
            else:
                cols = df.columns

            # Filter to numeric columns only
            numeric_cols = [
                c for c in cols
                if c not in ["ticker", "company_name", "sector", "industry",
                             "exchange", "country", "currency", "isin", "cusip",
                             "sedol", "f_fiscal_period", "f_fiscal_sector",
                             "f_fiscal_industry", "f_reported_currency", "f_cik",
                             "f_ticker", "s_split_date", "data_warning"]
            ]

            # If there's a selected row, prioritize its column
            options = [{"label": c, "value": c} for c in sorted(numeric_cols)]

            if selected_rows and len(selected_rows) > 0:
                col_involved = selected_rows[0].get("column_involved", "")
                # Extract first column if multiple
                if "/" in col_involved:
                    col_involved = col_involved.split("/")[0]
                if "," in col_involved:
                    col_involved = col_involved.split(",")[0].strip()
                if "(" in col_involved:
                    col_involved = col_involved.split("(")[0].strip()

                # Move the relevant column to the top
                if col_involved in numeric_cols:
                    options = [{"label": f">> {col_involved} (selected)", "value": col_involved}] + [
                        o for o in options if o["value"] != col_involved
                    ]

            return options

        return []

    @app.callback(
        Output("visualization-column-dropdown", "value"),
        Input("log-grid", "selectedRows"),
        State("ticker-dropdown", "value"),
    )
    def auto_select_column(selected_rows: Optional[list], ticker: Optional[str]):
        """Auto-select column based on selected log row."""
        if not selected_rows or not ticker:
            return None

        row = selected_rows[0]
        col_involved = row.get("column_involved", "")

        # Extract first column if multiple
        if "/" in col_involved:
            col_involved = col_involved.split("/")[0]
        if "," in col_involved:
            col_involved = col_involved.split(",")[0].strip()
        if "(" in col_involved:
            col_involved = col_involved.split("(")[0].strip()

        # Verify column exists
        if ticker in cleaned_dataframes:
            df = cleaned_dataframes[ticker]
            if isinstance(df, polars.LazyFrame):
                cols = df.collect_schema().names()
            else:
                cols = df.columns

            if col_involved in cols:
                return col_involved

        return None

    @app.callback(
        Output("error-chart", "figure"),
        [
            Input("ticker-dropdown", "value"),
            Input("visualization-column-dropdown", "value"),
            Input("log-grid", "selectedRows"),
            Input("error-category-dropdown", "value"),
        ],
    )
    def update_error_chart(
        ticker: Optional[str],
        column: Optional[str],
        selected_rows: Optional[list],
        category: Optional[str],
    ):
        """Update the error visualization chart."""
        fig = go.Figure()

        if not ticker or not column:
            fig.update_layout(
                title="Select a ticker and column to visualize",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white",
            )
            return fig

        # Get data
        original_df = original_dataframes.get(ticker)
        cleaned_df = cleaned_dataframes.get(ticker)

        if original_df is None or cleaned_df is None:
            fig.update_layout(
                title=f"No data available for {ticker}",
                template="plotly_white",
            )
            return fig

        # Collect if LazyFrame
        if isinstance(cleaned_df, polars.LazyFrame):
            cleaned_df = cleaned_df.collect()

        # Check if column exists
        if column not in original_df.columns or column not in cleaned_df.columns:
            fig.update_layout(
                title=f"Column '{column}' not found in data",
                template="plotly_white",
            )
            return fig

        # Determine date column
        date_col = "m_date" if "m_date" in original_df.columns else "f_filing_date"

        if date_col not in original_df.columns:
            fig.update_layout(
                title="No date column found in data",
                template="plotly_white",
            )
            return fig

        # Special handling for Accounting Equation Mismatch
        if category == "Accounting Mismatch" and selected_rows:
            row = selected_rows[0]
            if row.get("error_type") == "accounting_equation_mismatch":
                return _create_accounting_equation_chart(
                    original_df, cleaned_df, date_col, ticker, row
                )

        # Extract data
        try:
            dates = original_df[date_col].to_list()
            original_values = original_df[column].to_list()
            cleaned_values = cleaned_df[column].to_list()

            # Handle mismatched lengths by truncating to shorter
            min_len = min(len(dates), len(original_values), len(cleaned_values))
            if min_len < len(dates):
                dates = dates[:min_len]
                original_values = original_values[:min_len]
                cleaned_values = cleaned_values[:min_len]
        except Exception as e:
            fig.update_layout(
                title=f"Error extracting data: {e}",
                template="plotly_white",
            )
            return fig

        # Add original data (red dashed) - with None value handling
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=original_values,
                mode="lines",
                name="Original",
                line=dict(color="red", dash="dash", width=2),
                connectgaps=False,  # Don't connect across None values
            )
        )

        # Add cleaned data (green solid)
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=cleaned_values,
                mode="lines",
                name="Cleaned",
                line=dict(color="green", width=2),
                connectgaps=False,
            )
        )

        # Get error dates for highlighting - filter by column if possible
        error_dates = set()
        if ticker:
            try:
                ticker_logs_lf = normalized_logs_lf.filter(
                    polars.col("ticker") == ticker
                )
                if category:
                    ticker_logs_lf = ticker_logs_lf.filter(
                        polars.col("error_category") == category
                    )
                # Filter by column if applicable
                if column:
                    ticker_logs_col_lf = ticker_logs_lf.filter(
                        polars.col("column_involved").str.contains(column)
                    )
                    # Check if column filter has results
                    col_count = ticker_logs_col_lf.select(polars.len()).collect().item()
                    if col_count > 0:
                        ticker_logs_lf = ticker_logs_col_lf
                # Collect only dates column with limit
                error_dates = set(
                    ticker_logs_lf.select("date").limit(1000).collect()["date"].to_list()
                )
            except Exception:
                error_dates = set()

        # Convert dates list to strings for comparison
        dates_str_list = [serialize_date(d) for d in dates]

        # Add error date highlights as shapes (with small width for visibility)
        shapes = []
        for error_date in error_dates:
            if error_date and error_date in dates_str_list:
                # Find index to get actual date object for shape
                try:
                    idx = dates_str_list.index(error_date)
                    actual_date = dates[idx]
                    # Calculate a small offset for rectangle width (1 day equivalent)
                    shapes.append(
                        dict(
                            type="rect",
                            xref="x",
                            yref="paper",
                            x0=actual_date,
                            x1=actual_date,
                            y0=0,
                            y1=1,
                            fillcolor="rgba(255, 0, 0, 0.2)",
                            line=dict(width=2, color="rgba(255, 0, 0, 0.5)"),
                            layer="below",
                        )
                    )
                except (ValueError, IndexError):
                    pass

        # Auto-zoom to selected row's date if available
        xaxis_range = None
        if selected_rows and len(selected_rows) > 0:
            selected_date = selected_rows[0].get("date")
            if selected_date and selected_date in dates_str_list:
                # Find the index
                try:
                    idx = dates_str_list.index(selected_date)
                    # Show 30 data points before and after
                    start_idx = max(0, idx - 30)
                    end_idx = min(len(dates) - 1, idx + 30)
                    xaxis_range = [dates[start_idx], dates[end_idx]]
                except (ValueError, IndexError):
                    pass

        fig.update_layout(
            title=f"{ticker}: {column}",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            shapes=shapes,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        if xaxis_range:
            fig.update_xaxes(range=xaxis_range)

        return fig

    def _create_accounting_equation_chart(
        original_df: polars.DataFrame,
        cleaned_df: polars.DataFrame,
        date_col: str,
        ticker: str,
        selected_row: dict,
    ) -> go.Figure:
        """Create special chart for Accounting Equation Mismatch."""
        fig = go.Figure()

        # Required columns
        assets_col = "fbs_assets"
        liabs_col = "fbs_liabilities"
        equity_col = "fbs_stockholder_equity"
        nci_col = "fbs_noncontrolling_interest"

        required = [assets_col, liabs_col, equity_col]
        if not all(c in original_df.columns for c in required):
            fig.update_layout(
                title="Required columns not found for accounting equation chart",
                template="plotly_white",
            )
            return fig

        dates = original_df[date_col].to_list()
        assets = original_df[assets_col].to_list()

        # Calculate L + E + NCI
        liabilities = original_df[liabs_col].to_list()
        equity = original_df[equity_col].to_list()
        nci = original_df[nci_col].to_list() if nci_col in original_df.columns else [0] * len(dates)

        claims = [
            (l or 0) + (e or 0) + (n or 0)
            for l, e, n in zip(liabilities, equity, nci)
        ]

        # Add Assets line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=assets,
                mode="lines",
                name="Assets",
                line=dict(color="blue", width=2),
            )
        )

        # Add Claims (L+E+NCI) line
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=claims,
                mode="lines",
                name="Liabilities + Equity + NCI",
                line=dict(color="orange", width=2),
            )
        )

        # Highlight the selected error date
        selected_date = selected_row.get("date")
        shapes = []
        if selected_date:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=selected_date,
                    x1=selected_date,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(255, 0, 0, 0.3)",
                    line=dict(width=2, color="red"),
                    layer="below",
                )
            )

        fig.update_layout(
            title=f"{ticker}: Accounting Equation (Assets vs L+E+NCI)",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_white",
            shapes=shapes,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            hovermode="x unified",
        )

        return fig

    @app.callback(
        Output("toggle-false-positive-btn", "disabled"),
        Input("log-grid", "selectedRows"),
    )
    def enable_false_positive_button(selected_rows: Optional[list]):
        """Enable/disable the false positive toggle button."""
        return not (selected_rows and len(selected_rows) > 0)

    @app.callback(
        [
            Output("false-positive-status", "children"),
            Output("false-positives-store", "data", allow_duplicate=True),
        ],
        Input("toggle-false-positive-btn", "n_clicks"),
        State("log-grid", "selectedRows"),
        State("false-positives-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_false_positive(
        n_clicks: int,
        selected_rows: Optional[list],
        current_fp: Optional[dict],
    ):
        """Toggle false positive flag for selected row."""
        # Get current false positives
        fp_data = current_fp.copy() if current_fp else false_positives_container["data"].copy()

        if not selected_rows or len(selected_rows) == 0:
            return "No row selected", fp_data

        row = selected_rows[0]
        fp_key = get_false_positive_key(
            row.get("ticker", ""),
            row.get("date", ""),
            row.get("error_type", ""),
            row.get("column_involved", ""),
        )

        if fp_key in fp_data:
            del fp_data[fp_key]
            status = "Removed false positive flag"
        else:
            fp_data[fp_key] = {
                "ticker": row.get("ticker"),
                "date": row.get("date"),
                "error_type": row.get("error_type"),
                "column": row.get("column_involved"),
                "flagged_at": datetime.now().isoformat(),
            }
            status = "Marked as false positive"

        # Update container and save to file
        false_positives_container["data"] = fp_data
        save_false_positives(fp_data)

        return status, fp_data

    return app


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def run_dashboard(
    original_dataframes: dict[str, polars.DataFrame],
    cleaned_dataframes: dict[str, Union[polars.DataFrame, polars.LazyFrame]],
    logs: dict,
    debug: bool = True,
    port: int = 8050,
) -> None:
    """
    Run the Data Corrector dashboard.

    Args:
        original_dataframes: Dict of ticker -> original DataFrame
        cleaned_dataframes: Dict of ticker -> cleaned DataFrame/LazyFrame
        logs: Dict of log category -> log entries
        debug: Whether to run in debug mode
        port: Port number for the server
    """
    app = create_app(original_dataframes, cleaned_dataframes, logs)

    print(f"\n{'='*60}")
    print("Data Corrector Dashboard")
    print(f"{'='*60}")
    print(f"Starting server at http://127.0.0.1:{port}")
    print(f"Loaded {len(original_dataframes)} tickers")
    print(f"Debug mode: {debug}")
    print(f"{'='*60}\n")

    # Disable reloader to prevent re-running the entire data cleaning pipeline
    # The reloader spawns a child process that re-executes the script from the start
    app.run(debug=debug, port=port, use_reloader=False)
