import polars
import os
import glob
from typing import Dict, Union, Optional, List

data_dir = "../../Input/Data"
metadata_dir = "../../Input/Universe_Information/Universe_Information.csv"


# ============================================================================
# 1. DEFINE THE STRICT SCHEMA
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
    's_split_date_numerator': polars.Int32, 's_split_date_denominator': polars.Int32,
    'c_rank': polars.Int64, 'c_sector_rank': polars.Int64, 'c_industry_rank': polars.Int64,
    'c_count': polars.Int64, 'c_row_number': polars.Int64
}


# ============================================================================
# 2. FUNCTION IMPLEMENTATION
# ============================================================================

def read_csv_files_to_polars(
        directory_path: str,
        lazy: bool = True,
        max_files: Optional[int] = None,
        file_pattern: str = "*.csv",
        include_files: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        use_schema_override: bool = True
) -> Dict[str, Union[polars.LazyFrame, polars.DataFrame]]:
    """
    Reads CSV files from a directory into a dictionary of Polars frames.
    - Enforces a strict schema where known columns are typed correctly.
    - UNKNOWN columns are forced to Float64.
    - Handles US Date Formats (MM/DD/YYYY) by reading as String first, then parsing.
    """

    # 1. Locate files
    search_path = os.path.join(directory_path, file_pattern)
    all_files = glob.glob(search_path)

    if not all_files:
        print(f"No files found in {search_path}")
        return {}

    # 2. Filter files
    files_to_process = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        if include_files and filename not in include_files:
            continue
        if exclude_files and filename in exclude_files:
            continue
        files_to_process.append(file_path)

    # 3. Apply Limit
    if max_files is not None:
        files_to_process = files_to_process[:max_files]

    results = {}
    print(f"Processing {len(files_to_process)} files...")

    for file_path in files_to_process:
        filename = os.path.basename(file_path)

        try:
            # Step A: Create the Schema Overrides & conversion lists
            current_overrides = {}
            date_cols_to_convert = []

            if use_schema_override:
                # Scan header to see what columns exist in THIS file
                temp_scan = polars.scan_csv(file_path, n_rows=1)
                file_columns = temp_scan.collect_schema().names()

                for col in file_columns:
                    if col in FINANCIAL_DATA_SCHEMA:
                        target_type = FINANCIAL_DATA_SCHEMA[col]

                        # SPECIAL HANDLING: If it's a Date, read as String first to avoid crash
                        if target_type == polars.Date:
                            current_overrides[col] = polars.String
                            date_cols_to_convert.append(col)
                        else:
                            current_overrides[col] = target_type
                    else:
                        # Unknown column -> Default to Float64
                        current_overrides[col] = polars.Float64

            # Step B: Read the Data (Dates are read as Strings here)
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

            # Step C: Apply Date Conversion (String -> Date)
            if date_cols_to_convert:
                # We construct a list of expressions to run in parallel
                date_expressions = [
                    polars.col(c).str.to_date("%m/%d/%Y", strict=False).alias(c)
                    for c in date_cols_to_convert
                ]
                frame = frame.with_columns(date_expressions)

            results[filename] = frame

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return results