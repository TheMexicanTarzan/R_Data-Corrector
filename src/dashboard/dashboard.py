"""
Financial Data Corrector Dashboard

Interactive audit tool for visualizing and comparing original vs. cleaned financial data.
Supports inspection of error logs, visualization of corrections, and false positive flagging.
"""

import polars as pl
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, callback, ALL
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
from typing import Dict, List, Any, Union, Optional
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging


# ============================================================================
# LOG NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_logs(logs_dict: Dict[str, Any]) -> pl.DataFrame:
    """
    Normalize all error logs into a unified Polars DataFrame structure.

    Handles different log structures:
    - List of dicts (most functions)
    - Dict of lists (fill_negatives_fundamentals)
    - Nested dict (validate_financial_equivalencies)

    Args:
        logs_dict: Dictionary containing all error logs from sanity checks

    Returns:
        Unified Polars DataFrame with columns:
        - ticker, date, error_category, error_type, column,
          original_value, corrected_value, message, metadata
    """
    normalized_records = []

    # Debug: Print log structure
    print("\n=== DEBUG: Log Structure Analysis ===")
    for key, value in logs_dict.items():
        print(f"\n{key}:")
        print(f"  Type: {type(value)}")
        if isinstance(value, dict):
            print(f"  Keys: {list(value.keys())[:5]}...")  # First 5 keys
            if len(value) > 0:
                first_key = list(value.keys())[0]
                first_value = value[first_key]
                print(f"  First value type: {type(first_value)}")
                if isinstance(first_value, (list, dict)):
                    print(f"  First value length/keys: {len(first_value) if isinstance(first_value, (list, dict)) else 'N/A'}")
        elif isinstance(value, list):
            print(f"  Length: {len(value)}")
            if len(value) > 0:
                print(f"  First item type: {type(value[0])}")
    print("="*40 + "\n")

    # 1. Process unsorted_dates_logs (list of lists from parallel_process_tickers)
    if "unsorted_dates_logs" in logs_dict:
        unsorted_logs = logs_dict["unsorted_dates_logs"]

        # Handle list of log lists (one per ticker)
        if isinstance(unsorted_logs, list):
            for log_list in unsorted_logs:
                if not log_list or not isinstance(log_list, list):
                    continue
                for entry in log_list:
                    if not isinstance(entry, dict):
                        continue

                    ticker = entry.get("ticker", "UNKNOWN")
                    error_type = entry.get("error_type", "unknown")

                    if error_type == "order_mismatch":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("m_date") or entry.get("f_filing_date"),
                            "error_category": "Date Sorting",
                            "error_type": "order_mismatch",
                            "column": "date_order",
                            "original_value": str(entry.get("original_position")),
                            "corrected_value": str(entry.get("sorted_position")),
                            "message": f"Row moved from position {entry.get('original_position')} to {entry.get('sorted_position')}",
                            "metadata": json.dumps(entry)
                        })
                    elif error_type == "duplicates_removed":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": None,
                            "error_category": "Date Sorting",
                            "error_type": "duplicates_removed",
                            "column": "date",
                            "original_value": str(entry.get("duplicates_removed")),
                            "corrected_value": "0",
                            "message": f"{entry.get('duplicates_removed')} duplicates removed using {entry.get('strategy')} strategy",
                            "metadata": json.dumps(entry)
                        })
                    elif error_type == "no_date_columns":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": None,
                            "error_category": "Date Sorting",
                            "error_type": "no_date_columns",
                            "column": "date",
                            "original_value": None,
                            "corrected_value": None,
                            "message": "No date columns found for sorting",
                            "metadata": json.dumps(entry)
                        })

    # 2. Process negative_fundamentals_logs (list of dicts from parallel_process_tickers)
    if "negative_fundamentals_logs" in logs_dict:
        neg_fund_logs = logs_dict["negative_fundamentals_logs"]
        if isinstance(neg_fund_logs, list):
            for col_dict in neg_fund_logs:
                if not col_dict or not isinstance(col_dict, dict):
                    continue
                if not col_dict:
                    continue
                for column, entries in col_dict.items():
                    if not entries:
                        continue
                    for entry in entries:
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("m_date"),
                            "error_category": "Negative Fundamentals",
                            "error_type": "negative_value",
                            "column": column,
                            "original_value": str(entry.get(column)),
                            "corrected_value": "forward_filled",
                            "message": f"Negative value {entry.get(column)} in {column} replaced via forward fill",
                            "metadata": json.dumps(entry)
                        })

    # 3. Process negative_market_logs (list of lists from parallel_process_tickers)
    if "negative_market_logs" in logs_dict:
        neg_mkt_logs = logs_dict["negative_market_logs"]
        if isinstance(neg_mkt_logs, list):
            for log_list in neg_mkt_logs:
                if not log_list or not isinstance(log_list, list):
                    continue
                if not log_list:
                    continue
                for entry in log_list:
                    method = entry.get("method", "unknown")
                    normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                        "date": entry.get("date"),
                        "error_category": "Negative Market Data",
                        "error_type": "negative_value",
                        "column": entry.get("column"),
                        "original_value": str(entry.get("original_value")),
                        "corrected_value": str(entry.get("corrected_value")),
                        "message": f"Negative value corrected using {method}",
                        "metadata": json.dumps(entry)
                    })

    # 4. Process zero_wipeout_logs (list of lists from parallel_process_tickers)
    if "zero_wipeout_logs" in logs_dict:
        zero_logs = logs_dict["zero_wipeout_logs"]
        if isinstance(zero_logs, list):
            for log_list in zero_logs:
                if not log_list or not isinstance(log_list, list):
                    continue
                if not log_list:
                    continue
                for entry in log_list:
                    # Extract the columns that were zero
                    zero_cols = []
                    for key, val in entry.items():
                        if key not in ["ticker", "m_date", "m_volume"] and val == 0:
                            zero_cols.append(key)
    
                    normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                        "date": entry.get("m_date"),
                        "error_category": "Zero Wipeout",
                        "error_type": "zero_with_volume",
                        "column": ", ".join(zero_cols) if zero_cols else "shares",
                        "original_value": "0",
                        "corrected_value": "forward_filled",
                        "message": f"Zero values found with volume {entry.get('m_volume')}, replaced via forward fill",
                        "metadata": json.dumps(entry)
                    })
    
    # 5. Process shares_outstanding_logs (list of lists from parallel_process_tickers)
    if "shares_outstanding_logs" in logs_dict:
        shares_logs = logs_dict["shares_outstanding_logs"]
        if isinstance(shares_logs, list):
            for log_list in shares_logs:
                if not log_list or not isinstance(log_list, list):
                    continue
                if not log_list:
                    continue
                for entry in log_list:
                    # Extract column that had the 10x jump
                    columns_involved = [k for k in entry.keys()
                                       if k not in ["ticker", "m_date", "error_type"]]
    
                    normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                        "date": entry.get("m_date"),
                        "error_category": "Market Cap Scale",
                        "error_type": entry.get("error_type", "scale_10x_jump"),
                        "column": ", ".join(columns_involved),
                        "original_value": str(entry.get(columns_involved[0]) if columns_involved else None),
                        "corrected_value": "forward_filled",
                        "message": f"10x scale jump detected and corrected",
                        "metadata": json.dumps(entry)
                    })
    
    # 6. Process ohlc_logs (list of lists from parallel_process_tickers)
    if "ohlc_logs" in logs_dict:
        ohlc_logs_data = logs_dict["ohlc_logs"]
        if isinstance(ohlc_logs_data, list):
            for log_list in ohlc_logs_data:
                if not log_list or not isinstance(log_list, list):
                    continue
                if not log_list:
                    continue
                for entry in log_list:
                    error_type = entry.get("error_type", "unknown")
    
                    if error_type == "high_not_maximum":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("date"),
                            "error_category": "OHLC Integrity",
                            "error_type": "high_not_maximum",
                            "column": f"{entry.get('column_group', 'raw')}_high",
                            "original_value": str(entry.get("old_high")),
                            "corrected_value": str(entry.get("new_high")),
                            "message": entry.get("message", "High corrected to maximum"),
                            "metadata": json.dumps(entry)
                        })
                    elif error_type == "low_not_minimum":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("date"),
                            "error_category": "OHLC Integrity",
                            "error_type": "low_not_minimum",
                            "column": f"{entry.get('column_group', 'raw')}_low",
                            "original_value": str(entry.get("old_low")),
                            "corrected_value": str(entry.get("new_low")),
                            "message": entry.get("message", "Low corrected to minimum"),
                            "metadata": json.dumps(entry)
                        })
                    elif error_type == "vwap_outside_range":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("date"),
                            "error_category": "OHLC Integrity",
                            "error_type": "vwap_outside_range",
                            "column": f"{entry.get('column_group', 'raw')}_vwap",
                            "original_value": str(entry.get("old_vwap")),
                            "corrected_value": str(entry.get("new_vwap")),
                            "message": entry.get("message", "VWAP corrected to OHLC centroid"),
                            "metadata": json.dumps(entry)
                        })
    
    # 7. Process financial_unequivalencies_logs (list of nested dicts from parallel_process_tickers)
    if "financial_unequivalencies_logs" in logs_dict:
        fin_logs = logs_dict["financial_unequivalencies_logs"]
        if isinstance(fin_logs, list):
            for nested_dict in fin_logs:
                if not nested_dict or not isinstance(nested_dict, dict):
                    continue
                if not nested_dict:
                    continue
    
                # Hard filter errors (corrections)
                if "hard_filter_errors" in nested_dict:
                    for entry in nested_dict["hard_filter_errors"]:
                        error_type = entry.get("error_type", "unknown")
    
                        if error_type == "assets_mismatch":
                            normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                                "date": entry.get("date"),
                                "error_category": "Accounting Mismatch (Hard)",
                                "error_type": "assets_mismatch",
                                "column": "assets_components",
                                "original_value": f"Current: {entry.get('current')}, Noncurrent: {entry.get('noncurrent')}",
                                "corrected_value": f"Current: {entry.get('corrected_current')}, Noncurrent: {entry.get('corrected_noncurrent')}",
                                "message": f"Assets mismatch corrected using {entry.get('correction_method')} (diff: {entry.get('difference')})",
                                "metadata": json.dumps(entry)
                            })
                        elif error_type == "liabilities_mismatch":
                            normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                                "date": entry.get("date"),
                                "error_category": "Accounting Mismatch (Hard)",
                                "error_type": "liabilities_mismatch",
                                "column": "liabilities_components",
                                "original_value": f"Current: {entry.get('current')}, Noncurrent: {entry.get('noncurrent')}",
                                "corrected_value": f"Current: {entry.get('corrected_current')}, Noncurrent: {entry.get('corrected_noncurrent')}",
                                "message": f"Liabilities mismatch corrected using {entry.get('correction_method')} (diff: {entry.get('difference')})",
                                "metadata": json.dumps(entry)
                            })
    
                # Soft filter warnings (flags only)
                if "soft_filter_warnings" in nested_dict:
                    for entry in nested_dict["soft_filter_warnings"]:
                        error_type = entry.get("error_type", "unknown")
    
                        if error_type == "equity_mismatch":
                            normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                                "date": entry.get("date"),
                                "error_category": "Accounting Mismatch (Soft)",
                                "error_type": "equity_mismatch",
                                "column": "stockholder_equity",
                                "original_value": f"Total: {entry.get('total')}",
                                "corrected_value": "Not applicable",
                                "message": f"Equity mismatch flagged (diff: {entry.get('difference')})",
                                "metadata": json.dumps(entry)
                            })
                        elif error_type == "accounting_equation_mismatch":
                            normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                                "date": entry.get("date"),
                                "error_category": "Accounting Mismatch (Soft)",
                                "error_type": "accounting_equation_mismatch",
                                "column": "balance_sheet",
                                "original_value": f"Assets: {entry.get('assets (total)')}",
                                "corrected_value": "Not applicable",
                                "message": f"A ≠ L + E + NCI (diff: {entry.get('difference')})",
                                "metadata": json.dumps(entry)
                            })
                        elif error_type == "cash_mismatch":
                            normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                                "date": entry.get("date"),
                                "error_category": "Accounting Mismatch (Soft)",
                                "error_type": "cash_mismatch",
                                "column": "cash_equivalents",
                                "original_value": f"{list(entry.keys())[3]}: {list(entry.values())[3]}",
                                "corrected_value": "Not applicable",
                                "message": f"Cash equivalency mismatch flagged (diff: {entry.get('difference')})",
                                "metadata": json.dumps(entry)
                            })
    
    # 8. Process split_inconsistencies_logs (list of lists from parallel_process_tickers)
    if "split_inconsistencies_logs" in logs_dict:
        split_logs = logs_dict["split_inconsistencies_logs"]
        if isinstance(split_logs, list):
            for log_list in split_logs:
                if not log_list or not isinstance(log_list, list):
                    continue
                if not log_list:
                    continue
                for entry in log_list:
                    error_type = entry.get("error_type", "unknown")
    
                    if error_type == "price_split_mismatch":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("date"),
                            "error_category": "Split Consistency",
                            "error_type": "price_split_mismatch",
                            "column": entry.get("raw_column"),
                            "original_value": str(entry.get("original_adjusted_value")),
                            "corrected_value": str(entry.get("corrected_adjusted_value")),
                            "message": f"Split-adjusted price corrected (K_expected: {entry.get('k_expected')}, K_implied: {entry.get('k_implied')})",
                            "metadata": json.dumps(entry)
                        })
                    elif error_type == "volume_split_mismatch":
                        normalized_records.append({
                            "ticker": entry.get("ticker", "UNKNOWN"),
                            "date": entry.get("date"),
                            "error_category": "Split Consistency",
                            "error_type": "volume_split_mismatch",
                            "column": entry.get("raw_column"),
                            "original_value": str(entry.get("original_adjusted_value")),
                            "corrected_value": str(entry.get("corrected_adjusted_value")),
                            "message": f"Split-adjusted volume corrected (K_expected: {entry.get('k_expected')}, K_implied: {entry.get('k_implied')})",
                            "metadata": json.dumps(entry)
                        })
                    elif error_type in ["skipped_pair", "skipped_validation"]:
                        # These are informational, can be included or filtered
                        pass
    
    # Create Polars DataFrame
    if not normalized_records:
        # Return empty DataFrame with correct schema
        return pl.DataFrame({
            "ticker": [],
            "date": [],
            "error_category": [],
            "error_type": [],
            "column": [],
            "original_value": [],
            "corrected_value": [],
            "message": [],
            "metadata": [],
            "is_false_positive": []
        })

    df = pl.DataFrame(normalized_records)

    # Add false_positive column (default False)
    df = df.with_columns(pl.lit(False).alias("is_false_positive"))

    # Ensure date column is properly typed (handle multiple date formats)
    if "date" in df.columns:
        try:
            # Try standard format first
            df = df.with_columns(
                pl.col("date").cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias("date")
            )
        except Exception:
            try:
                # Try alternative format
                df = df.with_columns(
                    pl.col("date").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False).alias("date")
                )
            except Exception:
                # Leave as string if parsing fails
                df = df.with_columns(
                    pl.col("date").cast(pl.Utf8).alias("date")
                )

    return df


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_error_figure(message: str, message_type: str = "info") -> go.Figure:
    """
    Create a standardized error/info figure for the dashboard.

    Args:
        message: The message to display
        message_type: Type of message - "error", "warning", or "info"

    Returns:
        Plotly Figure object
    """
    color_map = {
        "error": "red",
        "warning": "orange",
        "info": "gray"
    }

    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=color_map.get(message_type, "gray"))
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        height=500
    )
    return fig


# ============================================================================
# DASHBOARD INITIALIZATION
# ============================================================================

def create_dashboard_app(
    original_dataframes: Dict[str, pl.DataFrame],
    cleaned_dataframes: Dict[str, pl.DataFrame],
    logs: Dict[str, Any]
) -> Dash:
    """Create and configure the Dash application."""

    # Normalize logs
    normalized_logs_df = normalize_logs(logs)

    # Note: For very large datasets (>10,000 errors or >100MB dataframes),
    # consider implementing server-side filtering or using a database backend
    # instead of storing all data in browser memory via dcc.Store

    # Initialize Dash app with Bootstrap theme
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
        suppress_callback_exceptions=True
    )

    # Handle empty logs case
    if len(normalized_logs_df) == 0:
        print("Warning: No error logs found. Dashboard will display empty state.")

    # Get unique tickers and error categories
    tickers = sorted(normalized_logs_df["ticker"].unique().to_list()) if len(normalized_logs_df) > 0 else []

    # Get categories and add unified "Accounting Mismatch" option
    raw_categories = sorted(normalized_logs_df["error_category"].unique().to_list()) if len(normalized_logs_df) > 0 else []
    categories = []
    has_hard = False
    has_soft = False

    for cat in raw_categories:
        if cat == "Accounting Mismatch (Hard)":
            has_hard = True
        elif cat == "Accounting Mismatch (Soft)":
            has_soft = True
        else:
            categories.append(cat)

    # Add unified accounting mismatch option if either hard or soft exists
    if has_hard or has_soft:
        categories.insert(0, "Accounting Mismatch")

    categories = sorted(categories)

    # ========================================================================
    # LAYOUT
    # ========================================================================

    app.layout = dbc.Container([
        # Store components for data persistence
        dcc.Store(id='normalized-logs-store', data=normalized_logs_df.to_dicts()),
        dcc.Store(id='original-dfs-store', data={ticker: df.to_dicts() for ticker, df in original_dataframes.items()}),
        dcc.Store(id='cleaned-dfs-store', data={ticker: df.to_dicts() for ticker, df in cleaned_dataframes.items()}),
        dcc.Store(id='false-positive-store', data=[]),

        # Header
        dbc.Row([
            dbc.Col([
                html.H1([
                    html.I(className="fa fa-chart-line me-3"),
                    "Data Corrector"
                ], className="text-primary mb-0"),
                html.P("Financial Data Cleaning Pipeline Audit Tool",
                       className="text-muted lead")
            ])
        ], className="mb-4 mt-3"),

        dbc.Row([
            # Sidebar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-filter me-2"),
                        html.Strong("Filters")
                    ]),
                    dbc.CardBody([
                        # Ticker Dropdown with search
                        html.Label("Ticker Symbol", className="fw-bold"),
                        dcc.Dropdown(
                            id='ticker-dropdown',
                            options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                            placeholder="Search ticker...",
                            searchable=True,
                            clearable=True,
                            className="mb-3"
                        ),

                        # Error Category Dropdown
                        html.Label("Error Category", className="fw-bold"),
                        dcc.Dropdown(
                            id='category-dropdown',
                            options=[{'label': cat, 'value': cat} for cat in categories],
                            placeholder="Select category...",
                            clearable=True,
                            className="mb-3"
                        ),

                        # Specific Error Type Dropdown (dynamic)
                        html.Label("Specific Error Type", className="fw-bold"),
                        dcc.Dropdown(
                            id='error-type-dropdown',
                            placeholder="Select error type...",
                            clearable=True,
                            className="mb-3"
                        ),

                        # Date Range (dynamic based on filtered data)
                        html.Label("Date Range", className="fw-bold"),
                        dcc.DatePickerRange(
                            id='date-range-picker',
                            className="mb-3",
                            style={"width": "100%"}
                        ),

                        # Toggle for Hard/Soft filters
                        html.Label("Filter Mode (for Accounting Mismatches)", className="fw-bold"),
                        dbc.RadioItems(
                            id='filter-mode-toggle',
                            options=[
                                {'label': 'Hard Errors (Corrected)', 'value': 'hard'},
                                {'label': 'Soft Warnings (Flagged)', 'value': 'soft'},
                                {'label': 'Both', 'value': 'both'}
                            ],
                            value='both',
                            className="mb-3"
                        ),

                        # Reset button
                        dbc.Button(
                            [html.I(className="fa fa-undo me-2"), "Reset Filters"],
                            id='reset-button',
                            color="secondary",
                            outline=True,
                            className="w-100"
                        )
                    ])
                ], className="shadow-sm")
            ], width=3),

            # Main Content
            dbc.Col([
                # Upper Section: Log Inspector
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-table me-2"),
                        html.Strong("Error Log Inspector"),
                        dbc.Badge(id="error-count-badge", className="ms-2", color="danger")
                    ]),
                    dbc.CardBody([
                        dag.AgGrid(
                            id='error-log-grid',
                            columnDefs=[
                                {"field": "ticker", "headerName": "Ticker", "width": 100, "pinned": "left"},
                                {"field": "date", "headerName": "Date", "width": 120},
                                {"field": "error_category", "headerName": "Category", "width": 180},
                                {"field": "error_type", "headerName": "Error Type", "width": 180},
                                {"field": "column", "headerName": "Column(s)", "width": 200},
                                {"field": "original_value", "headerName": "Original Value", "width": 200},
                                {"field": "corrected_value", "headerName": "Corrected Value", "width": 200},
                                {"field": "message", "headerName": "Message", "width": 300, "wrapText": True, "autoHeight": True},
                                {
                                    "field": "is_false_positive",
                                    "headerName": "False Positive",
                                    "width": 140,
                                    "cellRenderer": "agCheckboxCellRenderer",
                                    "editable": True,
                                    "cellStyle": {
                                        "styleConditions": [
                                            {
                                                "condition": "params.value === true",
                                                "style": {"backgroundColor": "#fff3cd", "fontWeight": "bold"}
                                            }
                                        ]
                                    }
                                }
                            ],
                            rowData=[],
                            columnSize="responsiveSizeToFit",
                            defaultColDef={
                                "sortable": True,
                                "filter": True,
                                "resizable": True
                            },
                            dashGridOptions={
                                "pagination": True,
                                "paginationPageSize": 20,
                                "rowSelection": "single",
                                "animateRows": True,
                                "rowClassRules": {
                                    "false-positive-row": "data.is_false_positive === true"
                                }
                            },
                            style={"height": "500px"}
                        )
                    ])
                ], className="shadow-sm mb-4"),

                # Lower Section: Visualization
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fa fa-chart-area me-2"),
                        html.Strong("Data Visualization"),
                        html.Span(id="chart-title", className="ms-3 text-muted")
                    ]),
                    dbc.CardBody([
                        dcc.Loading(
                            id="loading-chart",
                            type="default",
                            children=[
                                dcc.Graph(
                                    id='comparison-chart',
                                    style={"height": "500px"},
                                    config={'displayModeBar': True, 'displaylogo': False}
                                )
                            ]
                        )
                    ])
                ], className="shadow-sm")
            ], width=9)
        ])
    ], fluid=True, className="p-4", style={"backgroundColor": "#f8f9fa"})

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    @app.callback(
        Output('error-type-dropdown', 'options'),
        Output('error-type-dropdown', 'value'),
        Input('category-dropdown', 'value'),
        State('normalized-logs-store', 'data'),
        prevent_initial_call=True
    )
    def update_error_type_dropdown(selected_category, logs_data):
        """Update error type dropdown based on selected category."""
        if not selected_category or not logs_data:
            return [], None

        try:
            logs_df = pl.DataFrame(logs_data)
            filtered_df = logs_df.filter(pl.col("error_category") == selected_category)

            if len(filtered_df) == 0:
                return [], None

            error_types = sorted(filtered_df["error_type"].unique().to_list())
            options = [{'label': et, 'value': et} for et in error_types]

            return options, None
        except Exception as e:
            print(f"Error updating error type dropdown: {e}")
            return [], None

    @app.callback(
        Output('date-range-picker', 'start_date'),
        Output('date-range-picker', 'end_date'),
        Output('date-range-picker', 'min_date_allowed'),
        Output('date-range-picker', 'max_date_allowed'),
        Input('ticker-dropdown', 'value'),
        State('normalized-logs-store', 'data')
    )
    def update_date_range(selected_ticker, logs_data):
        """Update date range picker based on selected ticker."""
        if not selected_ticker or not logs_data:
            return None, None, None, None

        logs_df = pl.DataFrame(logs_data)
        filtered_df = logs_df.filter(
            (pl.col("ticker") == selected_ticker) &
            (pl.col("date").is_not_null())
        )

        if len(filtered_df) == 0:
            return None, None, None, None

        min_date = filtered_df["date"].min()
        max_date = filtered_df["date"].max()

        return min_date, max_date, min_date, max_date

    @app.callback(
        Output('error-log-grid', 'rowData'),
        Output('error-count-badge', 'children'),
        Input('ticker-dropdown', 'value'),
        Input('category-dropdown', 'value'),
        Input('error-type-dropdown', 'value'),
        Input('date-range-picker', 'start_date'),
        Input('date-range-picker', 'end_date'),
        Input('filter-mode-toggle', 'value'),
        State('normalized-logs-store', 'data'),
        State('false-positive-store', 'data')
    )
    def update_error_log_grid(ticker, category, error_type, start_date, end_date,
                              filter_mode, logs_data, false_positives):
        """Filter and display error logs based on selected criteria."""
        if not logs_data:
            return [], "0 errors"

        logs_df = pl.DataFrame(logs_data)

        # Apply filters
        filters = []

        if ticker:
            filters.append(pl.col("ticker") == ticker)

        if category:
            # Special handling for accounting mismatch categories
            if "Accounting Mismatch" in category:
                if filter_mode == "hard":
                    filters.append(pl.col("error_category") == "Accounting Mismatch (Hard)")
                elif filter_mode == "soft":
                    filters.append(pl.col("error_category") == "Accounting Mismatch (Soft)")
                else:  # both
                    filters.append(
                        (pl.col("error_category") == "Accounting Mismatch (Hard)") |
                        (pl.col("error_category") == "Accounting Mismatch (Soft)")
                    )
            else:
                filters.append(pl.col("error_category") == category)

        if error_type:
            filters.append(pl.col("error_type") == error_type)

        if start_date and end_date:
            try:
                # Ensure date column is proper date type for comparison
                start_date_obj = pl.lit(start_date).str.to_date() if isinstance(start_date, str) else pl.lit(start_date)
                end_date_obj = pl.lit(end_date).str.to_date() if isinstance(end_date, str) else pl.lit(end_date)

                filters.append(
                    (pl.col("date") >= start_date_obj) &
                    (pl.col("date") <= end_date_obj)
                )
            except Exception as e:
                print(f"Error filtering by date: {e}")

        # Apply all filters
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
            filtered_df = logs_df.filter(combined_filter)
        else:
            filtered_df = logs_df

        # Update false positive flags
        if false_positives:
            for fp in false_positives:
                filtered_df = filtered_df.with_columns(
                    pl.when(
                        (pl.col("ticker") == fp["ticker"]) &
                        (pl.col("date") == fp["date"]) &
                        (pl.col("error_type") == fp["error_type"])
                    )
                    .then(pl.lit(True))
                    .otherwise(pl.col("is_false_positive"))
                    .alias("is_false_positive")
                )

        # Convert to dict for AG Grid
        row_data = filtered_df.to_dicts()

        # Format dates as strings
        for row in row_data:
            if row.get("date"):
                row["date"] = str(row["date"])

        error_count = len(row_data)

        return row_data, f"{error_count} errors"

    @app.callback(
        Output('comparison-chart', 'figure'),
        Output('chart-title', 'children'),
        Input('error-log-grid', 'selectedRows'),
        State('ticker-dropdown', 'value'),
        State('original-dfs-store', 'data'),
        State('cleaned-dfs-store', 'data'),
        prevent_initial_call=False
    )
    def update_comparison_chart(selected_rows, selected_ticker, original_dfs_data, cleaned_dfs_data):
        """Generate comparison chart for selected error."""
        try:
            if not selected_rows or len(selected_rows) == 0:
                # Show empty chart with instruction
                return create_error_figure("Select a row from the error log to view visualization", "info"), ""

            selected_row = selected_rows[0]
            ticker = selected_row["ticker"]
            error_category = selected_row["error_category"]
            error_type = selected_row["error_type"]
            column = selected_row["column"]
            error_date = selected_row.get("date")

            # Get dataframes
            if ticker not in original_dfs_data or ticker not in cleaned_dfs_data:
                return create_error_figure(f"Data not available for ticker {ticker}", "error"), f"Error: Data not found for {ticker}"

            original_df = pl.DataFrame(original_dfs_data[ticker])
            cleaned_df = pl.DataFrame(cleaned_dfs_data[ticker])

            # Special handling for accounting_equation_mismatch
            if error_type == "accounting_equation_mismatch":
                fig, title = create_accounting_equation_chart(
                    original_df, cleaned_df, error_date, ticker
                )
                return fig, title

            # Standard dual-line chart
            # Determine which column to plot (handle comma-separated columns, take first)
            plot_column = column.split(",")[0].strip() if column else None

            # Validate column name
            if not plot_column:
                return create_error_figure("No column specified for visualization", "error"), "Error: No column"

            # Check if column exists
            if plot_column not in original_df.columns:
                return create_error_figure(f"Column '{plot_column}' not found in dataset", "error"), "Error: Column not found"

            # Create comparison chart
            fig = create_dual_line_chart(
                original_df, cleaned_df, plot_column, error_date, ticker, error_category
            )

            chart_title = f"{ticker} - {plot_column} ({error_category})"

            return fig, chart_title

        except Exception as e:
            logging.error(f"Error creating comparison chart: {e}")
            return create_error_figure(f"Error creating chart: {str(e)}", "error"), "Error"

    @app.callback(
        Output('false-positive-store', 'data'),
        Input('error-log-grid', 'cellValueChanged'),
        State('false-positive-store', 'data'),
        prevent_initial_call=True
    )
    def handle_false_positive_flag(cell_changed, false_positives):
        """Handle false positive checkbox changes."""
        try:
            if not cell_changed:
                return false_positives or []

            changed_data = cell_changed[0]
            row_data = changed_data.get('data', {})

            ticker = row_data.get('ticker')
            date = row_data.get('date')
            error_type = row_data.get('error_type')
            is_false_positive = row_data.get('is_false_positive', False)

            # Validation
            if not ticker or not error_type:
                return false_positives or []

            # Initialize if None
            if false_positives is None:
                false_positives = []

            # Create identifier
            fp_id = {"ticker": ticker, "date": date, "error_type": error_type}

            # Add or remove from list
            if is_false_positive:
                if fp_id not in false_positives:
                    false_positives.append(fp_id)
                    print(f"Flagged false positive: {ticker} - {error_type} on {date}")
            else:
                false_positives = [fp for fp in false_positives if fp != fp_id]
                print(f"Removed false positive flag: {ticker} - {error_type} on {date}")

            return false_positives

        except Exception as e:
            print(f"Error handling false positive flag: {e}")
            return false_positives or []

    @app.callback(
        Output('ticker-dropdown', 'value', allow_duplicate=True),
        Output('category-dropdown', 'value', allow_duplicate=True),
        Output('error-type-dropdown', 'value', allow_duplicate=True),
        Output('date-range-picker', 'start_date', allow_duplicate=True),
        Output('date-range-picker', 'end_date', allow_duplicate=True),
        Input('reset-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_filters(n_clicks):
        """Reset all filters to default values."""
        if n_clicks:
            return None, None, None, None, None
        return None, None, None, None, None

    return app


def create_dual_line_chart(
    original_df: pl.DataFrame,
    cleaned_df: pl.DataFrame,
    column: str,
    error_date: Optional[str],
    ticker: str,
    error_category: str
) -> go.Figure:
    """Create a dual-line comparison chart with error highlighting."""

    try:
        # Ensure dataframes are sorted by date
        date_col = "m_date" if "m_date" in original_df.columns else "f_filing_date"

        original_df = original_df.sort(date_col)
        cleaned_df = cleaned_df.sort(date_col)

        # Extract data
        dates_original = original_df[date_col].to_list()
        values_original = original_df[column].to_list()

        dates_cleaned = cleaned_df[date_col].to_list()
        values_cleaned = cleaned_df[column].to_list()

        # Create figure
        fig = go.Figure()

        # Add original data (red dashed line)
        fig.add_trace(go.Scatter(
            x=dates_original,
            y=values_original,
            mode='lines',
            name='Original',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='<b>Original</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
        ))

        # Add cleaned data (green solid line)
        fig.add_trace(go.Scatter(
            x=dates_cleaned,
            y=values_cleaned,
            mode='lines',
            name='Cleaned',
            line=dict(color='green', width=2),
            hovertemplate='<b>Cleaned</b><br>Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
        ))

        # Highlight error date with transparent red rectangle and auto-zoom
        x_range = None
        if error_date:
            try:
                error_date_parsed = datetime.strptime(error_date, "%Y-%m-%d").date()

                # Get y-axis range for the rectangle
                all_values = [v for v in values_original + values_cleaned if v is not None]
                if all_values:
                    y_min = min(all_values)
                    y_max = max(all_values)
                    y_range = y_max - y_min

                    # Add shaded region
                    fig.add_vrect(
                        x0=error_date_parsed,
                        x1=error_date_parsed,
                        fillcolor="red",
                        opacity=0.2,
                        layer="below",
                        line_width=2,
                        line_color="red"
                    )

                    # Auto-zoom: set x-axis range to ±30 days around error date
                    x_min = error_date_parsed - timedelta(days=30)
                    x_max = error_date_parsed + timedelta(days=30)
                    x_range = [x_min, x_max]

            except Exception:
                pass  # Skip if date parsing fails

        # Update layout
        layout_kwargs = {
            "title": f"{ticker} - {column}",
            "xaxis_title": "Date",
            "yaxis_title": column,
            "hovermode": 'x unified',
            "legend": dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            "template": "plotly_white",
            "height": 500
        }

        # Apply auto-zoom if error date exists
        if x_range:
            layout_kwargs["xaxis"] = dict(range=x_range, title="Date")

        fig.update_layout(**layout_kwargs)

        return fig

    except Exception as e:
        # Return error figure
        logging.error(f"Error creating dual-line chart: {e}")
        return create_error_figure(f"Error creating chart: {str(e)}", "error")


def create_accounting_equation_chart(
    original_df: pl.DataFrame,
    cleaned_df: pl.DataFrame,
    error_date: Optional[str],
    ticker: str
) -> tuple[go.Figure, str]:
    """
    Create special chart for accounting equation mismatch.
    Plots Assets vs. (Liabilities + Equity + NCI).
    """

    try:
        date_col = "m_date" if "m_date" in original_df.columns else "f_filing_date"

        # Required columns
        assets_col = "fbs_assets"
        liabs_col = "fbs_liabilities"
        equity_col = "fbs_stockholder_equity"
        nci_col = "fbs_noncontrolling_interest"

        # Check if columns exist
        required_cols = [assets_col, liabs_col, equity_col]
        if not all(col in original_df.columns for col in required_cols):
            return create_error_figure("Required accounting columns not found", "error"), "Error: Missing columns"

        # Sort by date
        original_df = original_df.sort(date_col)
        cleaned_df = cleaned_df.sort(date_col)

        # Calculate total claims (L + E + NCI)
        # Handle NCI (may be null)
        original_df = original_df.with_columns(
            (pl.col(liabs_col) + pl.col(equity_col) + pl.col(nci_col).fill_null(0.0)).alias("_total_claims")
        )

        cleaned_df = cleaned_df.with_columns(
            (pl.col(liabs_col) + pl.col(equity_col) + pl.col(nci_col).fill_null(0.0)).alias("_total_claims")
        )

        # Extract data
        dates = original_df[date_col].to_list()
        assets_original = original_df[assets_col].to_list()
        claims_original = original_df["_total_claims"].to_list()

        assets_cleaned = cleaned_df[assets_col].to_list()
        claims_cleaned = cleaned_df["_total_claims"].to_list()

        # Create figure
        fig = go.Figure()

        # Original data
        fig.add_trace(go.Scatter(
            x=dates,
            y=assets_original,
            mode='lines',
            name='Assets (Original)',
            line=dict(color='blue', dash='dash', width=2),
            hovertemplate='<b>Assets (Original)</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=claims_original,
            mode='lines',
            name='L + E + NCI (Original)',
            line=dict(color='red', dash='dash', width=2),
            hovertemplate='<b>L + E + NCI (Original)</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
        ))

        # Cleaned data
        fig.add_trace(go.Scatter(
            x=dates,
            y=assets_cleaned,
            mode='lines',
            name='Assets (Cleaned)',
            line=dict(color='blue', width=2),
            hovertemplate='<b>Assets (Cleaned)</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=dates,
            y=claims_cleaned,
            mode='lines',
            name='L + E + NCI (Cleaned)',
            line=dict(color='green', width=2),
            hovertemplate='<b>L + E + NCI (Cleaned)</b><br>Date: %{x}<br>Value: %{y:,.0f}<extra></extra>'
        ))

        # Highlight error date
        if error_date:
            try:
                error_date_parsed = datetime.strptime(error_date, "%Y-%m-%d").date()
                fig.add_vrect(
                    x0=error_date_parsed,
                    x1=error_date_parsed,
                    fillcolor="orange",
                    opacity=0.3,
                    layer="below",
                    line_width=2,
                    line_color="orange"
                )
            except Exception:
                pass

        # Update layout
        fig.update_layout(
            title=f"{ticker} - Accounting Equation: Assets = Liabilities + Equity + NCI",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template="plotly_white",
            height=500
        )

        return fig, f"{ticker} - Accounting Equation Check"

    except Exception as e:
        # Return error figure
        logging.error(f"Error creating accounting equation chart: {e}")
        return create_error_figure(f"Error creating chart: {str(e)}", "error"), "Error"


# ============================================================================
# ENTRY POINT
# ============================================================================

def run_dashboard(
    original_dataframes: Dict[str, Union[pl.DataFrame, pl.LazyFrame]],
    cleaned_dataframes: Dict[str, Union[pl.DataFrame, pl.LazyFrame]],
    logs: Dict[str, Any],
    debug: bool = True,
    port: int = 8050
):
    """
    Launch the Data Corrector dashboard.

    Args:
        original_dataframes: Dictionary of ticker -> original DataFrame/LazyFrame
        cleaned_dataframes: Dictionary of ticker -> cleaned DataFrame/LazyFrame
        logs: Dictionary of error logs from sanity checks
        debug: Run in debug mode (default: True)
        port: Port to run dashboard on (default: 8050)
    """

    # Convert LazyFrames to DataFrames
    original_dfs_collected = {}
    for ticker, df in original_dataframes.items():
        if isinstance(df, pl.LazyFrame):
            original_dfs_collected[ticker] = df.collect()
        else:
            original_dfs_collected[ticker] = df

    cleaned_dfs_collected = {}
    for ticker, df in cleaned_dataframes.items():
        if isinstance(df, pl.LazyFrame):
            cleaned_dfs_collected[ticker] = df.collect()
        else:
            cleaned_dfs_collected[ticker] = df

    # Create app
    app = create_dashboard_app(
        original_dfs_collected,
        cleaned_dfs_collected,
        logs
    )

    # Run server
    print(f"\n{'='*80}")
    print(f"=� Data Corrector Dashboard is running on http://localhost:{port}")
    print(f"{'='*80}\n")

    app.run_server(debug=debug, port=port, host='0.0.0.0')
