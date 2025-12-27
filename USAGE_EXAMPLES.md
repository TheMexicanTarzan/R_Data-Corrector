# validate_financial_equivalencies - Usage Examples

## Basic Usage

### Single DataFrame Validation
```python
import polars
from src.modules.errors.sanity_check.sanity_check import validate_financial_equivalencies

# Load data
df = polars.read_csv("financial_data.csv")

# Validate and clean
cleaned_df, error_log = validate_financial_equivalencies(
    df=df,
    ticker="AAPL",
    date_col="m_date",
    tolerance=1.0
)

# Access results
print(f"Rows with warnings: {cleaned_df.filter(polars.col('data_warning')).height}")
print(f"Hard errors corrected: {len(error_log['hard_filter_errors'])}")
print(f"Soft warnings flagged: {len(error_log['soft_filter_warnings'])}")
```

### LazyFrame Validation (Recommended for Large Datasets)
```python
import polars

# Load as LazyFrame for better performance
lf = polars.scan_csv("financial_data.csv")

# Validate (remains lazy until collected)
cleaned_lf, error_log = validate_financial_equivalencies(
    df=lf,
    ticker="AAPL",
    date_col="m_date",
    tolerance=0.01  # Stricter tolerance
)

# Process further or collect when needed
result = cleaned_lf.collect()
```

## Integration with Parallel Processing

### Process Multiple Tickers in Parallel
```python
from src.features.lazy_parallelization import parallel_process_tickers
from src.modules.errors.sanity_check.sanity_check import validate_financial_equivalencies

# Dictionary of ticker → LazyFrame
data_dict = {
    "AAPL": polars.scan_csv("data/AAPL.csv"),
    "MSFT": polars.scan_csv("data/MSFT.csv"),
    "GOOGL": polars.scan_csv("data/GOOGL.csv"),
}

# Process in parallel
cleaned_data, all_logs = parallel_process_tickers(
    data_dict=data_dict,
    columns=[""],  # Not used by this function, but required
    function=validate_financial_equivalencies,
    max_workers=8,
    show_progress=True
)

# Access cleaned data
aapl_cleaned = cleaned_data["AAPL"]

# Aggregate error logs
total_hard_errors = sum(len(log.get("hard_filter_errors", [])) for log in all_logs)
total_soft_warnings = sum(len(log.get("soft_filter_warnings", [])) for log in all_logs)
```

## Advanced Error Analysis

### Export Hard Filter Corrections to CSV
```python
import polars

# Run validation
cleaned_df, error_log = validate_financial_equivalencies(
    df=df,
    ticker="AAPL",
    date_col="m_date"
)

# Convert hard filter errors to DataFrame
if error_log["hard_filter_errors"]:
    errors_df = polars.DataFrame(error_log["hard_filter_errors"])
    errors_df.write_csv("hard_filter_corrections.csv")

    # Analyze correction magnitude
    errors_df = errors_df.with_columns([
        (polars.col("corrected_current") - polars.col("current")).abs().alias("current_adjustment"),
        (polars.col("corrected_noncurrent") - polars.col("noncurrent")).abs().alias("noncurrent_adjustment")
    ])

    print(errors_df.select([
        "date",
        "error_type",
        "difference",
        "current_adjustment",
        "noncurrent_adjustment",
        "correction_method"
    ]))
```

### Analyze Soft Filter Warnings
```python
# Extract soft warnings
if error_log["soft_filter_warnings"]:
    warnings_df = polars.DataFrame(error_log["soft_filter_warnings"])

    # Separate by warning type
    equity_warnings = warnings_df.filter(polars.col("error_type") == "equity_mismatch")
    cash_warnings = warnings_df.filter(polars.col("error_type") == "cash_mismatch")

    print(f"Equity mismatches: {equity_warnings.height}")
    print(f"Cash mismatches: {cash_warnings.height}")

    # Find largest discrepancies
    if cash_warnings.height > 0:
        largest_cash_discrepancy = cash_warnings.sort("difference", descending=True).head(1)
        print("\nLargest cash discrepancy:")
        print(largest_cash_discrepancy)
```

## Custom Tolerance Settings

### Strict Validation (Penny-Perfect)
```python
# Very strict tolerance for audited financials
cleaned_df, error_log = validate_financial_equivalencies(
    df=df,
    ticker="AAPL",
    tolerance=0.01  # 1 cent tolerance
)
```

### Relaxed Validation (Millions)
```python
# Relaxed tolerance for preliminary data in millions
cleaned_df, error_log = validate_financial_equivalencies(
    df=df,
    ticker="AAPL",
    tolerance=1000.0  # $1,000 tolerance (if data is in dollars)
)
```

## Filtering and Inspection

### Inspect Only Flagged Rows
```python
# Get cleaned data
cleaned_df, _ = validate_financial_equivalencies(df, "AAPL")

# Filter to rows with warnings
flagged_rows = cleaned_df.filter(polars.col("data_warning") == True)

print(f"Found {flagged_rows.height} rows with data quality warnings")
print(flagged_rows.select([
    "m_date",
    "fbs_stockholder_equity",
    "fbs_common_stock_value",
    "fbs_additional_paid_in_capital",
    "fbs_retained_earnings",
    "fbs_other_stockholder_equity"
]))
```

### Compare Before/After Correction
```python
import polars

# Keep original for comparison
original_df = df.clone()

# Validate and clean
cleaned_df, error_log = validate_financial_equivalencies(
    df=df,
    ticker="AAPL"
)

# Compare specific columns that were corrected
if error_log["hard_filter_errors"]:
    comparison = original_df.select([
        "m_date",
        polars.col("fbs_current_assets").alias("original_current_assets"),
        polars.col("fbs_noncurrent_assets").alias("original_noncurrent_assets")
    ]).join(
        cleaned_df.select([
            "m_date",
            polars.col("fbs_current_assets").alias("corrected_current_assets"),
            polars.col("fbs_noncurrent_assets").alias("corrected_noncurrent_assets")
        ]),
        on="m_date"
    )

    # Show only rows that changed
    changed = comparison.filter(
        (polars.col("original_current_assets") != polars.col("corrected_current_assets")) |
        (polars.col("original_noncurrent_assets") != polars.col("corrected_noncurrent_assets"))
    )

    print("Corrected rows:")
    print(changed)
```

## Pipeline Integration

### Sequential Validation Pipeline
```python
from src.modules.errors.sanity_check.sanity_check import (
    sort_dates,
    fill_negatives_fundamentals,
    validate_financial_equivalencies,
    ohlc_integrity
)

# Define pipeline
def financial_data_pipeline(df, ticker):
    """Complete data cleaning pipeline"""
    logs = {}

    # Step 1: Sort and deduplicate
    df, sort_logs = sort_dates(
        df, ticker, df.columns,
        dedupe_strategy="latest"
    )
    logs["sort"] = sort_logs

    # Step 2: Fill negative fundamentals
    df, neg_logs = fill_negatives_fundamentals(
        df,
        columns=["fbs_assets", "fbs_liabilities"],
        ticker=ticker
    )
    logs["negatives"] = neg_logs

    # Step 3: Validate financial equivalencies
    df, val_logs = validate_financial_equivalencies(
        df, ticker
    )
    logs["validation"] = val_logs

    # Step 4: OHLC integrity (if market data present)
    if "m_high" in df.columns:
        df, ohlc_logs = ohlc_integrity(df, ticker)
        logs["ohlc"] = ohlc_logs

    return df, logs

# Run full pipeline
cleaned_df, all_logs = financial_data_pipeline(df, "AAPL")
```

## Monitoring and Alerts

### Alert on High Error Rates
```python
def validate_with_alerts(df, ticker, max_error_rate=0.05):
    """Validate data and alert if error rate exceeds threshold"""
    cleaned_df, error_log = validate_financial_equivalencies(df, ticker)

    total_rows = df.height
    hard_errors = len(error_log["hard_filter_errors"])
    soft_warnings = len(error_log["soft_filter_warnings"])

    hard_error_rate = hard_errors / total_rows if total_rows > 0 else 0
    soft_warning_rate = soft_warnings / total_rows if total_rows > 0 else 0

    if hard_error_rate > max_error_rate:
        print(f"⚠️  WARNING: {ticker} has {hard_error_rate:.1%} hard errors (threshold: {max_error_rate:.1%})")

    if soft_warning_rate > max_error_rate:
        print(f"⚠️  WARNING: {ticker} has {soft_warning_rate:.1%} soft warnings (threshold: {max_error_rate:.1%})")

    return cleaned_df, error_log

# Use with alert threshold
cleaned_df, error_log = validate_with_alerts(df, "AAPL", max_error_rate=0.10)
```

## Export Logs for Auditing

### Create Audit Trail
```python
import json
from datetime import datetime

def create_audit_log(ticker, error_log, output_dir="audit_logs"):
    """Export detailed audit log to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/{ticker}_{timestamp}_audit.json"

    audit_data = {
        "ticker": ticker,
        "timestamp": timestamp,
        "summary": {
            "hard_errors": len(error_log["hard_filter_errors"]),
            "soft_warnings": len(error_log["soft_filter_warnings"])
        },
        "details": error_log
    }

    with open(filename, "w") as f:
        json.dump(audit_data, f, indent=2, default=str)

    print(f"Audit log saved to: {filename}")
    return filename

# Create audit trail
cleaned_df, error_log = validate_financial_equivalencies(df, "AAPL")
audit_file = create_audit_log("AAPL", error_log)
```
