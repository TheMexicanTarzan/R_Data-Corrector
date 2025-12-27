# Financial Data Cleaning Audit Dashboard

A robust, interactive Dash application to visualize the results of the financial data cleaning pipeline. This dashboard serves as an audit tool for comparing original vs. cleaned data and inspecting error logs.

## Features

### 1. Log Normalizer
- Unifies different log structures from various cleaning functions into a standardized Polars DataFrame
- Handles all error types:
  - Negative Fundamentals (`fill_negatives_fundamentals`)
  - Negative Market Data (`fill_negatives_market`)
  - Zero Wipeout (`zero_wipeout`)
  - Market Cap Scale Errors (`mkt_cap_scale_error`)
  - OHLC Integrity (`ohlc_integrity`)
  - Financial Equivalencies (`validate_financial_equivalencies`)
  - Date Sorting Issues (`sort_dates`)

### 2. Interactive Dashboard Layout

#### Sidebar Filters
- **Ticker Dropdown**: Select from available tickers
- **Error Category Dropdown**: Filter by cleaning function category
- **Specific Error Dropdown**: Dynamic dropdown based on selected category
- **Financial Equivalencies Toggle**: Switch between Hard Errors, Soft Warnings, or All

#### Log Inspector (AG Grid)
- Displays error logs with columns:
  - Ticker
  - Date
  - Error Type
  - Column(s) Involved
  - Original Value
  - Corrected Value
  - Message
  - Severity
  - False Positive Flag
- Sortable and filterable columns
- Pagination support
- Row selection for detailed inspection

#### Time Series Comparator
- **Dual-Line Chart**:
  - Original data (red dashed line)
  - Cleaned data (green solid line)
- **Error Markers**: Orange "X" markers on dates with errors
- **Interactive Zoom**: Clicking a log entry auto-zooms to ±30 days around the error date
- **Special Charts**:
  - For accounting equation mismatches, displays Assets vs (Liabilities + Equity + NCI)
  - Shows difference plot to visualize gaps

### 3. False Positive Flagging
- Users can flag errors as false positives
- Flags are stored in `false_positives.json` for future reference
- Can clear false positive flags
- Persistent storage across sessions

### 4. Special Handling for Financial Equivalencies

The dashboard provides special handling for `validate_financial_equivalencies`:

- **Hard Filter Errors**: Structural corrections (Assets, Liabilities decomposition)
  - Shows correction method (proportional scaling vs residual plug)
  - Displays before/after values

- **Soft Filter Warnings**: Non-corrected warnings
  - Equity component mismatches
  - Cash equivalency checks
  - Fundamental accounting equation (A = L + E + NCI)

- **Accounting Equation Chart**:
  - Top panel: Assets line vs Total Claims (L + E + NCI) line
  - Bottom panel: Difference over time
  - Helps visualize the gap and correction impact

## Installation

```bash
cd src/dashboard
pip install -r requirements.txt
```

## Usage

### With Dummy Data (Testing)

Run the dashboard with auto-generated test data:

```bash
python dashboard.py
```

This will:
1. Generate fake financial data for 3 tickers (AAPL, GOOGL, MSFT)
2. Inject various types of errors
3. Run simplified cleaning
4. Launch the dashboard at http://localhost:8050

### With Real Data (Production)

```python
from pathlib import Path
import polars as pl
from src.dashboard.dashboard import FinancialDashboard, LogNormalizer
from src.data_corrector import (
    dataframe_dict,
    dataframe_dict_clean_financial_equivalencies,
    negative_fundamentals_logs,
    negative_market_logs,
    zero_wipeout_logs,
    shares_outstanding_logs,
    ohlc_logs,
    financial_unequivalencies_logs,
    unsorted_dates_logs
)

# Collect original and cleaned dataframes
original_dfs = {ticker: df.collect() for ticker, df in dataframe_dict.items()}
cleaned_dfs = {ticker: df.collect() for ticker, df in dataframe_dict_clean_financial_equivalencies.items()}

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

# Create and run dashboard
dashboard = FinancialDashboard(original_dfs, cleaned_dfs, all_logs)
dashboard.run(debug=False, port=8050)
```

## Dashboard Components

### LogNormalizer Class
Handles normalization of different log formats:
- `normalize_logs()`: Main entry point
- `_normalize_negative_fundamentals()`: Handles dict-of-lists format
- `_normalize_negative_market()`: Handles list-of-dicts format
- `_normalize_zero_wipeout()`: Handles zero wipeout logs
- `_normalize_mkt_cap_scale()`: Handles scale error logs
- `_normalize_ohlc_integrity()`: Handles OHLC validation logs
- `_normalize_financial_equivalencies()`: Handles hard/soft filter logs
- `_normalize_sort_dates()`: Handles date sorting logs

### DummyDataGenerator Class
Generates realistic test data:
- `generate_ticker_data()`: Creates fake financial data
- `_inject_errors()`: Adds various error types
- `_simple_clean()`: Simulates cleaning and log generation

### FinancialDashboard Class
Main application:
- `__init__()`: Initialize with dataframes and logs
- `_build_layout()`: Constructs Dash UI
- `_register_callbacks()`: Sets up interactivity
- `_create_accounting_equation_chart()`: Special chart for accounting equations
- `run()`: Launches the server

## Callbacks

1. **update_error_dropdown**: Dynamically populates error types based on category
2. **update_log_table**: Filters and displays logs based on selections
3. **update_timeseries**: Generates time series chart with error markers
4. **handle_false_positive**: Manages false positive flags

## Data Flow

```
Original Data → Cleaning Functions → Logs (Various Formats)
                                          ↓
                                  LogNormalizer
                                          ↓
                              Unified DataFrame
                                          ↓
                              Dashboard Filters
                                          ↓
                    Log Inspector ←→ Time Series Chart
                                          ↓
                              False Positive Flagging
```

## File Structure

```
src/dashboard/
├── dashboard.py           # Main dashboard application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── false_positives.json  # False positive flags (created at runtime)
```

## Technical Details

### Performance
- Uses Polars for fast data manipulation
- Lazy evaluation where possible
- Efficient filtering with boolean masks
- Pagination for large log tables

### Log Storage Schema

Normalized logs have the following schema:

```python
{
    "ticker": str,           # Ticker symbol
    "date": str,             # Date of error (YYYY-MM-DD)
    "error_category": str,   # High-level category
    "error_type": str,       # Specific error type
    "column": str,           # Column(s) involved
    "original_value": float, # Original value
    "corrected_value": float,# Corrected value (None if not corrected)
    "message": str,          # Human-readable message
    "severity": str,         # info/warning/error
    "false_positive": bool   # False positive flag
}
```

## Customization

### Adding New Error Types

To add support for a new cleaning function:

1. Add a new normalization method in `LogNormalizer`:
```python
@staticmethod
def _normalize_your_function(logs: List[Dict]) -> List[Dict]:
    rows = []
    # Parse your log format
    return rows
```

2. Add it to `normalize_logs()`:
```python
elif category == "your_category":
    normalized_rows.extend(
        LogNormalizer._normalize_your_function(logs)
    )
```

3. Update `all_logs` dictionary when instantiating the dashboard

### Changing Chart Styles

Modify the `update_timeseries` callback in `_register_callbacks()`:
- Line colors: `line=dict(color="your_color")`
- Marker styles: `marker=dict(symbol="your_symbol")`
- Layout: `fig.update_layout(...)`

## Troubleshooting

### Dashboard won't load
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify port 8050 is not in use
- Check console for error messages

### No data displayed
- Verify that logs are not empty
- Check ticker names match between dataframes and logs
- Ensure date formats are consistent (YYYY-MM-DD)

### Charts not updating
- Verify callback connections
- Check browser console for JavaScript errors
- Ensure selected ticker has data

## Future Enhancements

- [ ] Export filtered logs to CSV/Excel
- [ ] Batch false positive flagging
- [ ] Custom date range selector
- [ ] Multi-ticker comparison charts
- [ ] Statistical summary dashboard
- [ ] Email alerts for severe errors
- [ ] Integration with database storage
- [ ] User authentication and permissions

## License

Part of the R_Data-Corrector project.
