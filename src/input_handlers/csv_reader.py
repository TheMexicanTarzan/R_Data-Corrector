import polars
import os
import glob
from pathlib import Path
from typing import Union, Optional, Tuple

# Get the directory where this module is located
_MODULE_DIR = Path(__file__).parent.resolve()
# Default paths relative to the project root (assuming src/input_handlers/ structure)
_PROJECT_ROOT = _MODULE_DIR.parent.parent
data_dir = str(_PROJECT_ROOT / "Input" / "Data")
metadata_dir = str(_PROJECT_ROOT / "Input" / "Universe_Information" / "Universe_Information.csv")

# ============================================================================
# 1. DEFINE SCHEMAS
# ============================================================================

FINANCIAL_DATA_SCHEMA = {
    # --- Market Data ---
    'm_open': polars.Float64, 'm_high': polars.Float64, 'm_low': polars.Float64, 'm_close': polars.Float64,
    'm_volume': polars.Int64, 'm_vwap': polars.Float64, 'm_transactions': polars.Int64,
    'm_date': polars.Date,

    # --- Split Adjusted ---
    'm_open_split_adjusted': polars.Float64, 'm_high_split_adjusted': polars.Float64,
    'm_low_split_adjusted': polars.Float64, 'm_close_split_adjusted': polars.Float64,
    'm_volume_split_adjusted': polars.Int64, 'm_vwap_split_adjusted': polars.Float64,

    # --- Dividend & Split Adjusted ---
    'm_open_dividend_and_split_adjusted': polars.Float64, 'm_high_dividend_and_split_adjusted': polars.Float64,
    'm_low_dividend_and_split_adjusted': polars.Float64, 'm_close_dividend_and_split_adjusted': polars.Float64,
    'm_volume_dividend_and_split_adjusted': polars.Int64, 'm_vwap_dividend_and_split_adjusted': polars.Float64,

    # --- Identifiers ---
    'ticker': polars.String, 'company_name': polars.String, 'sector': polars.String, 'industry': polars.String,
    'exchange': polars.String, 'country': polars.String, 'currency': polars.String,
    'isin': polars.String, 'cusip': polars.String, 'sedol': polars.String,
    'f_fiscal_period': polars.String, 'f_fiscal_sector': polars.String, 'f_fiscal_industry': polars.String,
    'f_reported_currency': polars.String, 'f_cik': polars.String, 'f_ticker': polars.String,

    # --- Dates (Specific overrides) ---
    'f_filing_date': polars.String, 'f_period_end_date': polars.String,
    'd_declaration_date': polars.String, 'd_ex_dividend_date': polars.String,
    'd_record_date': polars.String, 'd_payment_date': polars.String,
    's_split_date': polars.String, 'f_accepted_date': polars.String,

    # --- Integers (Specific overrides) ---
    'f_fiscal_year': polars.Int32, 'f_fiscal_quarter': polars.Int32, 'd_frequency': polars.Int32,
    's_split_date_numerator': polars.Float32, 's_split_date_denominator': polars.Float32,
    'c_rank': polars.Int64, 'c_sector_rank': polars.Int64, 'c_industry_rank': polars.Int64,
    'c_count': polars.Int64, 'c_row_number': polars.Int64
}

METADATA_SCHEMA = {
    'symbol': polars.String,
    'price': polars.Float64,
    'marketCap': polars.Float64,
    'beta': polars.Float64,
    'lastDividend': polars.Float64,
    'range': polars.String,
    'change': polars.Float64,
    'changePercentage': polars.Float64,
    'volume': polars.Int64,
    'averageVolume': polars.Int64,
    'companyName': polars.String,
    'currency': polars.String,
    'cik': polars.String,
    'isin': polars.String,
    'cusip': polars.String,
    'exchangeFullName': polars.String,
    'exchange': polars.String,
    'industry': polars.String,
    'website': polars.String,
    'description': polars.String,
    'ceo': polars.String,
    'sector': polars.String,
    'country': polars.String,
    'fullTimeEmployees': polars.String,  # Often contains commas or is null
    'phone': polars.String,
    'address': polars.String,
    'city': polars.String,
    'state': polars.String,
    'zip': polars.String,
    'image': polars.String,
    'ipoDate': polars.String,  # Read as string first to handle formats safely
    'defaultImage': polars.Boolean,
    'isEtf': polars.Boolean,
    'isActivelyTrading': polars.Boolean,
    'isAdr': polars.Boolean,
    'isFund': polars.Boolean
}


# ============================================================================
# 2. FUNCTION IMPLEMENTATION
# ============================================================================

def read_csv_files_to_polars(
        directory_path: str,
        metadata_path: str = metadata_dir,
        lazy: bool = True,
        max_files: Optional[int] = None,
        file_pattern: str = "*.csv",
        include_files: Optional[list[str]] = None,
        exclude_files: Optional[list[str]] = None,
        use_schema_override: bool = True
) -> dict[str, Tuple[Union[polars.LazyFrame, polars.DataFrame], polars.LazyFrame]]:
    """
    Reads CSV files from a directory into a dictionary of Polars frames.

    Returns:
        dict: Keys are filenames. Values are Tuples: (TickerData, Metadata)
              - TickerData: LazyFrame or DataFrame of price history
              - Metadata: LazyFrame containing the single row of info for that ticker
    """

    # 1. Prepare Metadata
    # We scan the metadata file once. We will filter it later for each ticker.
    try:
        # Note: We infer schema length 0 to force using our strict schema,
        # or we can allow inference if the file is messy. Here we enforce schema overrides.
        full_metadata_lf = polars.scan_csv(
            metadata_path,
            schema_overrides=METADATA_SCHEMA,
            infer_schema_length=10000
        )

        # Convert ipoDate from String to Date if needed
        full_metadata_lf = full_metadata_lf.with_columns(
            polars.col("ipoDate").str.to_date("%Y-%m-%d", strict=False).alias("ipoDate")
        )
    except Exception as e:
        print(f"Warning: Could not load metadata from {metadata_path}. Error: {e}")
        # Create an empty dummy LF if metadata fails, to prevent crash
        full_metadata_lf = polars.LazyFrame(schema=METADATA_SCHEMA)

    # 2. Locate Data Files
    search_path = os.path.join(directory_path, file_pattern)
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"No files found in {search_path}")
        return {}

    # 3. Filter files
    files_to_process = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if include_files and filename not in include_files:
            continue
        if exclude_files and filename in exclude_files:
            continue
        files_to_process.append(file_path)

    # 4. Apply Limit
    if max_files is not None:
        files_to_process = files_to_process[:max_files]

    results = {}
    print(f"Processing {len(files_to_process)} files...")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)

        # Extract ticker symbol from filename (e.g., "AAPL.csv" -> "AAPL")
        ticker_symbol = os.path.splitext(filename)[0]

        try:
            # --- PART A: Process Financial Data ---
            current_overrides = {}
            date_cols_to_convert = []

            if use_schema_override:
                temp_scan = polars.scan_csv(file_path, n_rows=1)
                file_columns = temp_scan.collect_schema().names()

                for col in file_columns:
                    if col in FINANCIAL_DATA_SCHEMA:
                        target_type = FINANCIAL_DATA_SCHEMA[col]
                        if target_type == polars.Date:
                            current_overrides[col] = polars.String
                            date_cols_to_convert.append(col)
                        else:
                            current_overrides[col] = target_type
                    else:
                        current_overrides[col] = polars.Float64

            if lazy:
                frame = polars.scan_csv(
                    file_path,
                    schema_overrides=current_overrides,
                    infer_schema_length=10000
                )
            else:
                frame = polars.read_csv(
                    file_path,
                    schema_overrides=current_overrides,
                    infer_schema_length=10000
                )

            if date_cols_to_convert:
                date_expressions = [
                    polars.col(c).str.to_date("%m/%d/%Y", strict=False).alias(c)
                    for c in date_cols_to_convert
                ]
                frame = frame.with_columns(date_expressions)

            # --- PART B: Store Result with Full Metadata ---
            # Pass the full metadata LazyFrame so filters can query peer tickers
            # (e.g., mahalanobis_filter needs to find all tickers in the same sector)
            results[filename] = (frame, full_metadata_lf)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results