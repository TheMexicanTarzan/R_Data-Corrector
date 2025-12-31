# Data Corrector Dashboard

An interactive Dash application for visualizing and auditing the financial data cleaning pipeline.

## Overview

The Data Corrector Dashboard provides a comprehensive interface for:
- **Inspecting error logs** from data cleaning operations
- **Comparing original vs. cleaned data** through interactive visualizations
- **Flagging false positives** for quality control
- **Filtering errors** by ticker, category, type, and date range

## Features

### 📊 Error Log Inspector
- **AG Grid table** with sorting, filtering, and pagination
- Displays error details including:
  - Ticker symbol
  - Date of error
  - Error category and type
  - Affected column(s)
  - Original and corrected values
  - Descriptive message
  - False positive flag checkbox

### 📈 Interactive Visualizations
- **Dual-line charts** comparing original (red dashed) vs. cleaned (green solid) data
- **Auto-zoom** to ±30 days around error date when clicking a row
- **Error date highlighting** with transparent red overlay
- **Special accounting equation chart** for A = L + E + NCI mismatches

### 🔍 Advanced Filtering
- **Ticker search** with searchable dropdown
- **Error category** filter
- **Specific error type** filter (dynamic based on category)
- **Date range** picker
- **Hard/Soft filter toggle** for accounting mismatches
  - Hard: Errors that were corrected
  - Soft: Warnings that were flagged but not modified

### ✅ False Positive Flagging
- Mark errors as false positives using checkbox
- Visual highlighting of flagged rows
- Persistent storage across dashboard session

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `polars` - Fast DataFrame library
- `numpy` - Numerical computing
- `scipy` - Scientific computing
- `dash` - Web application framework
- `dash-bootstrap-components` - Bootstrap components for Dash
- `dash-ag-grid` - AG Grid integration for Dash
- `plotly` - Interactive charting library

## Usage

The dashboard is automatically launched after the data cleaning pipeline completes:

```python
python -m src.data_corrector
```

This will:
1. Run the full data cleaning pipeline
2. Generate error logs
3. Launch the dashboard on `http://localhost:8050`

### Manual Launch

You can also launch the dashboard programmatically:

```python
from src.dashboard.dashboard import run_dashboard

run_dashboard(
    original_dataframes=original_dfs,  # Dict[ticker, DataFrame]
    cleaned_dataframes=cleaned_dfs,    # Dict[ticker, DataFrame]
    logs=error_logs,                    # Dict of error logs
    debug=True,                         # Enable debug mode
    port=8050                           # Port number
)
```

## Dashboard Layout

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Corrector                            │
│           Financial Data Cleaning Pipeline Audit Tool        │
├───────────────┬─────────────────────────────────────────────┤
│   SIDEBAR     │         MAIN CONTENT                         │
│               │                                               │
│  Filters:     │  Error Log Inspector (AG Grid)               │
│  • Ticker     │  ┌────────────────────────────────────────┐  │
│  • Category   │  │ Ticker│Date│Category│Type│Column│...  │  │
│  • Error Type │  │ AAPL  │... │OHLC    │... │high  │...  │  │
│  • Date Range │  └────────────────────────────────────────┘  │
│  • Filter Mode│                                               │
│               │  Data Visualization (Plotly)                 │
│  [Reset]      │  ┌────────────────────────────────────────┐  │
│               │  │      Original vs Cleaned Chart          │  │
│               │  │  🔴 Original (dashed)                   │  │
│               │  │  🟢 Cleaned (solid)                     │  │
│               │  │  🟥 Error date highlight                │  │
│               │  └────────────────────────────────────────┘  │
└───────────────┴─────────────────────────────────────────────┘
```

## Error Categories

### 1. **Date Sorting**
- Order mismatches
- Duplicate removals
- Missing date columns

### 2. **Negative Fundamentals**
- Negative values in fundamental columns
- Forward-fill corrections

### 3. **Negative Market Data**
- Negative values in market price/volume data
- Cubic spline or last-valid-value corrections

### 4. **Zero Wipeout**
- Zero share values with positive volume
- Forward-fill corrections

### 5. **Market Cap Scale**
- 10x scale jumps in shares/market cap
- Forward-fill corrections

### 6. **OHLC Integrity**
- High not maximum violation
- Low not minimum violation
- VWAP outside [Low, High] range

### 7. **Accounting Mismatch (Hard)**
- Assets = Current + Noncurrent (corrected via proportional scaling)
- Liabilities = Current + Noncurrent (corrected via proportional scaling)

### 8. **Accounting Mismatch (Soft)**
- Equity component mismatch (flagged only)
- Cash equivalency mismatch (flagged only)
- Accounting equation A = L + E + NCI (flagged only)

### 9. **Split Consistency**
- Price split adjustment mismatches
- Volume split adjustment mismatches

## How to Use

### Basic Workflow

1. **Launch the dashboard** after data cleaning completes
2. **Select a ticker** from the dropdown to filter errors for a specific company
3. **Choose an error category** to narrow down the type of issues
4. **Optionally select a specific error type** for granular filtering
5. **Click a row** in the Error Log Inspector to view the visualization
6. **Review the chart** to see original vs. cleaned data
   - Red dashed line = Original data
   - Green solid line = Cleaned data
   - Red highlighted area = Date where error occurred
7. **Flag false positives** by checking the "False Positive" box

### Advanced Filtering

#### Filter by Date Range
1. Select a ticker
2. Use the date range picker to focus on a specific time period
3. All errors outside this range will be hidden

#### Hard vs. Soft Accounting Mismatches
For accounting-related errors, use the filter mode toggle:
- **Hard Errors**: Show only errors that were automatically corrected
- **Soft Warnings**: Show only flagged discrepancies (no correction applied)
- **Both**: Show all accounting mismatches

### Understanding the Visualizations

#### Standard Dual-Line Chart
- Shows the time series of a specific column
- Compares original (before correction) vs. cleaned (after correction)
- Error date is highlighted and auto-zoomed

#### Accounting Equation Chart
- Special visualization for A = L + E + NCI errors
- Shows 4 lines:
  - Assets (Original) - Blue dashed
  - Assets (Cleaned) - Blue solid
  - L + E + NCI (Original) - Red dashed
  - L + E + NCI (Cleaned) - Green solid
- Gap between Assets and Claims indicates the violation

## Architecture

### Log Normalization
The `normalize_logs()` function unifies 8 different error log structures into a single Polars DataFrame:

```python
{
    "ticker": str,
    "date": date,
    "error_category": str,
    "error_type": str,
    "column": str,
    "original_value": str,
    "corrected_value": str,
    "message": str,
    "metadata": json str,
    "is_false_positive": bool
}
```

### Error Log Structures Handled

1. **List of dicts** - Most sanity check functions
2. **Dict of lists** - `fill_negatives_fundamentals` (keyed by column)
3. **Nested dict** - `validate_financial_equivalencies` (hard/soft keys)

### Callbacks

The dashboard uses Dash callbacks for interactivity:

1. **update_error_type_dropdown** - Populates error types based on selected category
2. **update_date_range** - Sets date picker bounds based on ticker
3. **update_error_log_grid** - Filters and displays error rows
4. **update_comparison_chart** - Generates visualization for selected row
5. **handle_false_positive_flag** - Persists false positive flags
6. **reset_filters** - Clears all filter selections

## Performance Considerations

### Data Size Limits
- **Recommended**: <10,000 errors, <100MB total dataframe size
- For larger datasets:
  - Consider implementing server-side filtering
  - Use a database backend (PostgreSQL, DuckDB)
  - Implement data pagination/lazy loading

### Memory Usage
- All dataframes are stored in browser memory via `dcc.Store`
- Each ticker's original and cleaned data is converted to dicts
- Error logs are normalized into a single DataFrame

### Optimization Tips
1. Filter by ticker early to reduce data loaded in charts
2. Use date range filtering for large time series
3. Close unused tabs to free browser memory
4. For production use, consider migrating to a server-side data store

## Troubleshooting

### Dashboard doesn't load
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify port 8050 is not in use
- Check console for Python errors

### No data appears
- Ensure data cleaning pipeline completed successfully
- Verify error logs were generated
- Check browser console for JavaScript errors

### Charts don't display
- Ensure the selected ticker exists in both original and cleaned dataframes
- Verify the column exists in the dataset
- Check for null/NaN values in the data

### False positive flags don't persist
- Flags are stored in browser session only
- Refreshing the page will reset flags
- For persistent storage, implement a database backend

## Future Enhancements

Potential improvements for production use:

1. **Database Backend** - PostgreSQL/DuckDB for large datasets
2. **Export Functionality** - Download filtered logs as CSV/Excel
3. **User Authentication** - Multi-user support with saved sessions
4. **False Positive Persistence** - Save flags to database
5. **Comparison Mode** - Side-by-side comparison of multiple tickers
6. **Statistical Summary** - Dashboard-level statistics and charts
7. **Automated Reports** - Generate PDF/HTML audit reports
8. **Real-time Updates** - WebSocket integration for live data
9. **Custom Annotations** - Add notes/comments to specific errors
10. **Audit Trail** - Track who flagged what as false positive

## Support

For issues, questions, or contributions:
- Review the code documentation in `src/dashboard/dashboard.py`
- Check the sanity check function implementations in `src/modules/errors/sanity_check/sanity_check.py`
- Consult the main data corrector pipeline in `src/data_corrector.py`

## License

[Add your license information here]
