import polars
from typing import Union, Literal
import logging

MAX_LOGS = 10


def _add_quarter_column(
        df: Union[polars.DataFrame, polars.LazyFrame],
        date_col: str = "m_date"
) -> Union[polars.DataFrame, polars.LazyFrame]:
    """
    Add a quarter identifier column to the dataframe based on the date column.

    Args:
        df: Input DataFrame or LazyFrame
        date_col: Name of the date column to extract quarter from

    Returns:
        DataFrame/LazyFrame with added '_quarter' column in format 'YYYY-QN'
    """
    # Get schema to check column type
    is_lazy = isinstance(df, polars.LazyFrame)
    schema = df.collect_schema() if is_lazy else df.schema
    col_dtype = schema.get(date_col)

    # Parse date if it's a string
    if col_dtype == polars.String or col_dtype == polars.Utf8:
        date_expr = polars.col(date_col).str.to_date("%m/%d/%Y", strict=False)
    else:
        date_expr = polars.col(date_col).cast(polars.Date, strict=False)

    # Create quarter identifier: YYYY-Q1, YYYY-Q2, etc.
    quarter_expr = (
        date_expr.dt.year().cast(polars.Utf8) +
        "-Q" +
        date_expr.dt.quarter().cast(polars.Utf8)
    ).alias("_quarter")

    return df.with_columns(quarter_expr)


def sort_dates(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        dedupe_strategy: Literal["earliest", "latest", "all"] = "all",
        shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Sort a polars DataFrame or LazyFrame by date columns with a defined hierarchy.

    Args:
        df: The polars DataFrame or LazyFrame to sort.
        ticker: The ticker symbol for logging/tracking purposes.
        columns: List of column names present in the dataframe.
        date_col: The fallback date column name (default: 'm_date').
        dedupe_strategy: How to handle duplicate entries in date_col.
            - "earliest": Keep entry with earliest f_filing_date.
            - "latest": Keep entry with latest f_filing_date.
            - "all": Keep all duplicate entries.

    Returns:
        A tuple containing:
            - The sorted DataFrame or LazyFrame
            - A list of dicts documenting order mismatches and removed duplicates
    """
    date_hierarchy = [
        date_col,
        "f_filing_date",
    ]

    sort_columns = [col for col in date_hierarchy if col in columns]
    logs = []

    if not sort_columns:
        logs.append(
            {
                "ticker": ticker,
                "error_type": "no_date_columns",
                "message": "No date columns found for sorting",
            }
        )
        return df, logs

    is_lazy = isinstance(df, polars.LazyFrame)
    primary_sort_col = sort_columns[0]
    temp_sort_cols = [f"_sort_{col}" for col in sort_columns]

    # Get schema to check column types (lazy-safe)
    schema = df.collect_schema() if is_lazy else df.schema

    # Build cast expressions based on column type
    cast_expressions = []
    for col in sort_columns:
        col_dtype = schema.get(col)
        if col_dtype == polars.String or col_dtype == polars.Utf8:
            expr = polars.col(col).str.to_date("%m/%d/%Y", strict=False).alias(f"_sort_{col}")
        else:
            expr = polars.col(col).cast(polars.Date, strict=False).alias(f"_sort_{col}")
        cast_expressions.append(expr)

    # MEMORY FIX: Keep everything lazy until final result
    # Add original index and cast date columns
    working_lf = (
        df.lazy()
        .with_row_index("_original_idx")
        .with_columns(cast_expressions)
    )

    # Create a lazy version that's sorted by temp_sort_cols where date is valid
    # Use a single sort operation on the whole frame, nulls will naturally go to end
    sorted_lf = working_lf.sort(by=temp_sort_cols, nulls_last=True)

    # MEMORY FIX: Sample mismatches for logging instead of collecting all
    # Only collect a limited sample for audit logging (avoid memory explosion)
    MAX_MISMATCH_LOGS = 100

    # To detect mismatches, we need to compare positions - collect only for logging
    # Use a separate lightweight query for mismatch detection
    try:
        mismatch_sample = (
            working_lf
            .with_columns(
                polars.col("_original_idx").rank("ordinal").over(
                    polars.col(temp_sort_cols[0]).is_not_null()
                ).alias("_expected_rank")
            )
            .sort(by=temp_sort_cols, nulls_last=True)
            .with_row_index("_sorted_idx")
            .filter(
                (polars.col("_original_idx") != polars.col("_sorted_idx")) &
                polars.col(primary_sort_col).is_not_null()
            )
            .select(["_original_idx", "_sorted_idx"] + sort_columns)
            .limit(MAX_MISMATCH_LOGS)
            .collect()
        )

        if mismatch_sample.height > 0:
            mismatch_dicts = mismatch_sample.to_dicts()
            for row in mismatch_dicts:
                row["ticker"] = ticker
                row["error_type"] = "order_mismatch"
                row["original_position"] = row.pop("_original_idx")
                row["sorted_position"] = row.pop("_sorted_idx")
            logs.extend(mismatch_dicts)
    except Exception:
        # If mismatch detection fails, continue without logging
        pass

    # The actual sorted result - stays lazy
    result_df = sorted_lf.drop(["_original_idx"] + temp_sort_cols)

    if dedupe_strategy != "all" and date_col in columns:
        declaration_col = "f_filing_date"
        if declaration_col in columns:
            # MEMORY FIX: Keep deduplication lazy
            # Create temporary sort columns for deduplication sorting
            dedupe_sort_cols = [date_col, declaration_col]
            temp_dedupe_cols = [f"_sort_{col}" for col in dedupe_sort_cols]
            dedupe_cast_exprs = []
            for col in dedupe_sort_cols:
                col_dtype = schema.get(col)
                if col_dtype == polars.String or col_dtype == polars.Utf8:
                    expr = polars.col(col).str.to_date("%m/%d/%Y", strict=False).alias(f"_sort_{col}")
                else:
                    expr = polars.col(col).cast(polars.Date, strict=False).alias(f"_sort_{col}")
                dedupe_cast_exprs.append(expr)

            if dedupe_strategy == "earliest":
                result_df = (
                    result_df.lazy()
                    .with_columns(dedupe_cast_exprs)
                    .sort(by=temp_dedupe_cols)
                    .drop(temp_dedupe_cols)
                    .unique(subset=[date_col], keep="first", maintain_order=True)
                )
            elif dedupe_strategy == "latest":
                result_df = (
                    result_df.lazy()
                    .with_columns(dedupe_cast_exprs)
                    .sort(by=temp_dedupe_cols, descending=[False, True])
                    .drop(temp_dedupe_cols)
                    .unique(subset=[date_col], keep="first", maintain_order=True)
                )

            # MEMORY FIX: Final sort stays lazy - no need for complex position tracking
            result_df = (
                result_df
                .with_columns(cast_expressions)
                .sort(by=temp_sort_cols, nulls_last=True)
                .drop(temp_sort_cols)
            )

            # Log approximate duplicate count (sample-based to avoid full collection)
            logs.append(
                {
                    "ticker": ticker,
                    "error_type": "duplicates_removed",
                    "strategy": dedupe_strategy,
                    "duplicates_removed": "deduplication_applied",
                }
            )

    # MEMORY FIX: Result is already lazy, convert to DataFrame only if input was eager
    if not is_lazy:
        result_df = result_df.collect()

    return result_df, logs


def fill_negatives_fundamentals(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        columns: list[str],
        ticker: str,
        date_col: str = 'm_date',
        shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], dict[str, list[dict]]]:
    """
    Replaces negative values in specified columns with the last non-negative value.
    Processes data quarterly - evaluates once per quarter and applies corrections to all rows in that quarter.
    Works with both DataFrame (eager) and LazyFrame (lazy) execution.

    Args:
        df: Input polars DataFrame or LazyFrame.
        columns: List of column names to clean.
        ticker: String representing the ticker symbol (e.g., 'AAPL').
        date_col: Name of the column containing dates.

    Returns:
        tuple: (Cleaned DataFrame/LazyFrame, Audit Dictionary)
        - Returns same type as input (DataFrame → DataFrame, LazyFrame → LazyFrame)
        - Audit Dictionary format:
        {
            'column_name': [
                {'ticker': 'AAPL', 'quarter': 'YYYY-QN', 'date_range_start': '...',
                 'date_range_end': '...', 'value': -5, 'count': N},
                ...
            ]
        }

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
        Fundamental data is quarterly, so each quarter is evaluated once.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    audit_log = {}

    # MEMORY FIX: Limit audit log entries per column
    MAX_AUDIT_PER_COLUMN = MAX_LOGS

    # Add quarter column for grouping
    df_with_quarter = _add_quarter_column(df, date_col)

    # 1. Audit Step - group by quarter and log once per quarter
    for col in columns:
        # Build query for problem quarters (group by quarter, take first negative value per quarter)
        problem_query = (
            df_with_quarter.lazy()
            .filter(polars.col(col) < 0)
            .group_by("_quarter")
            .agg([
                polars.col(date_col).min().alias("date_range_start"),
                polars.col(date_col).max().alias("date_range_end"),
                polars.col(col).first().alias("value"),  # Representative value from quarter
                polars.col(col).count().alias("count")  # Number of negative entries in quarter
            ])
            .sort("date_range_start")
            .limit(MAX_AUDIT_PER_COLUMN)
        )

        # Collect only the limited problem quarters
        problem_quarters = problem_query.collect()

        # Check if there are any problem quarters
        if len(problem_quarters) > 0:
            # Convert to dicts and inject the ticker string into every entry
            entries = problem_quarters.to_dicts()
            for entry in entries:
                entry['ticker'] = ticker
                # Rename _quarter to quarter for cleaner output
                entry['quarter'] = entry.pop('_quarter')
            audit_log[col] = entries

    # 2. Cleaning Step - works lazily or eagerly
    # Apply forward fill within each quarter group
    cleaned_df = df_with_quarter.with_columns(
        (
            polars.when(polars.col(col) >= 0)
            .then(polars.col(col))
            .otherwise(None)
            .forward_fill()
            .alias(col)
        )
        for col in columns
    ).drop("_quarter")

    return cleaned_df, audit_log


def zero_wipeout(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        columns: list[str],
        ticker: str,
        date_col: str = 'm_date',
        shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Identifies rows where share columns are 0 AND 'm_volume' > 0.
    These 0 values are replaced via forward fill.
    Processes data quarterly - evaluates once per quarter and applies corrections to all rows in that quarter.
    Works with both DataFrame (eager) and LazyFrame (lazy) execution.

    Args:
        df: Input polars DataFrame or LazyFrame.
        columns: List of column names to check for zero wipeout.
        ticker: String representing the ticker symbol.
        date_col: Name of the column containing dates.

    Returns:
        tuple: (Cleaned DataFrame/LazyFrame, Audit List)
        - Audit List format: [{'ticker': 'AAPL', 'quarter': 'YYYY-QN',
                               'date_range_start': '...', 'date_range_end': '...',
                               'affected_columns': [...], 'count': N}, ...]

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
        Fundamental data is quarterly, so each quarter is evaluated once.
    """
    import polars as pl

    is_lazy = isinstance(df, pl.LazyFrame)
    target_cols = columns

    # MEMORY FIX: Limit audit log entries
    MAX_AUDIT_ENTRIES = MAX_LOGS

    # Add quarter column for grouping
    df_with_quarter = _add_quarter_column(df, date_col)

    # Build condition: ANY target column is 0 AND volume > 0
    # Create condition for each column, then combine with OR
    conditions = [
        (pl.col(col) == 0) for col in target_cols
    ]
    # Combine: (col1 == 0 OR col2 == 0) AND volume > 0
    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition = combined_condition | cond

    final_condition = combined_condition & (pl.col("m_volume") > 0)

    # 1. Audit Step - group by quarter and log once per quarter
    # For each column, identify which quarters have the issue
    problem_query = (
        df_with_quarter.lazy()
        .filter(final_condition)
        .group_by("_quarter")
        .agg([
            pl.col(date_col).min().alias("date_range_start"),
            pl.col(date_col).max().alias("date_range_end"),
            pl.col("m_volume").first().alias("volume"),  # Representative volume
            pl.len().alias("count")  # Number of affected rows in quarter
        ] + [
            # Track which columns had zeros
            pl.col(col).min().alias(f"{col}_min") for col in target_cols
        ])
        .sort("date_range_start")
        .limit(MAX_AUDIT_ENTRIES)
    )

    # Collect only the limited problem quarters
    problem_quarters = problem_query.collect()

    audit_log = []
    if len(problem_quarters) > 0:
        entries = problem_quarters.to_dicts()
        for entry in entries:
            entry['ticker'] = ticker
            entry['quarter'] = entry.pop('_quarter')
            # Identify which columns had zeros
            affected_cols = [col for col in target_cols if entry.get(f"{col}_min") == 0]
            entry['affected_columns'] = affected_cols
            # Clean up temporary min columns
            for col in target_cols:
                entry.pop(f"{col}_min", None)
        audit_log = entries

    # 2. Cleaning Step - process each column separately
    # For each target column, replace 0s (when volume > 0) with None, then forward fill
    for col in target_cols:
        # Condition for this specific column
        col_condition = (pl.col(col) == 0) & (pl.col("m_volume") > 0)

        df_with_quarter = df_with_quarter.with_columns(
            pl.when(col_condition)
            .then(None)  # Replace problematic 0s with null
            .otherwise(pl.col(col))  # Keep everything else
            .forward_fill()  # Fill nulls with previous values
            .alias(col)
        )

    # Remove quarter column before returning
    result_df = df_with_quarter.drop("_quarter")

    return result_df, audit_log


def mkt_cap_scale_error(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str] = [""],
        date_col: str = 'm_date',
        market_cap_col: str = 'c_market_cap',
        shares_outstanding_col: str = 'fis_weighted_average_diluted_shares_outstanding',
        shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Identifies and corrects rows where share columns jump by 10x or more
    compared to the previous quarter. If both market cap and shares outstanding
    jump together, forward fills the entire quarter.
    Processes data quarterly - evaluates once per quarter and applies corrections to all rows in that quarter.

    Args:
        df: Input polars DataFrame or LazyFrame.
        ticker: String representing the ticker symbol.
        columns: List of column names to check for scale errors.
        date_col: Name of the column containing dates.
        market_cap_col: Name of the market cap column.
        shares_outstanding_col: Name of the shares outstanding column.

    Returns:
        tuple: (Cleaned DataFrame/LazyFrame, Audit List)
        - Audit List format: [{'ticker': 'AAPL', 'quarter': 'YYYY-QN',
                               'date_range_start': '...', 'date_range_end': '...',
                               'error_type': '...', 'affected_columns': [...], 'count': N}, ...]

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
        Fundamental data is quarterly, so each quarter is evaluated once.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df.lazy()
    target_cols = columns

    # Get schema to check column existence (lazy-safe)
    schema_cols = set(working_lf.collect_schema().names())

    # Check if both market cap and shares outstanding columns exist
    has_market_cap = market_cap_col in schema_cols
    has_shares = shares_outstanding_col in schema_cols

    # Add quarter column for grouping
    working_lf = _add_quarter_column(working_lf, date_col)

    # Build combined condition for audit - check for 10x jumps from previous row
    conditions = []
    available_target_cols = []
    for col in target_cols:
        if col in schema_cols:
            available_target_cols.append(col)
            prev_shares = polars.col(col).shift(1)
            col_condition = (polars.col(col) >= (prev_shares * 10))
            conditions.append(col_condition)

    if not conditions:
        return df, []

    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition = combined_condition | cond

    # MEMORY FIX: Limit audit log entries
    MAX_AUDIT_ENTRIES = MAX_LOGS

    audit_log = []
    try:
        # Group by quarter and identify affected quarters
        problem_query = (
            working_lf
            .filter(combined_condition)
            .group_by("_quarter")
            .agg([
                polars.col(date_col).min().alias("date_range_start"),
                polars.col(date_col).max().alias("date_range_end"),
                polars.len().alias("count")
            ] + [
                polars.col(col).first().alias(f"{col}_value") for col in available_target_cols
            ])
            .sort("date_range_start")
            .limit(MAX_AUDIT_ENTRIES)
        )

        problem_quarters = problem_query.collect()

        if problem_quarters.height > 0:
            entries = problem_quarters.to_dicts()
            for entry in entries:
                entry['ticker'] = ticker
                entry['quarter'] = entry.pop('_quarter')
                entry['error_type'] = "scale_10x_jump"
                # Identify which columns had jumps
                affected_cols = [col for col in available_target_cols if f"{col}_value" in entry]
                entry['affected_columns'] = affected_cols
                # Clean up temporary value columns
                for col in available_target_cols:
                    entry.pop(f"{col}_value", None)
            audit_log = entries
            logging.info(f"Found scale errors (10x jump) for ticker {ticker} in {len(audit_log)} quarters")
    except Exception:
        pass  # Continue without audit if it fails

    # MEMORY FIX: Use lazy evaluation for correlated jump detection
    # Instead of collecting and iterating, use pure Polars expressions
    if has_market_cap and has_shares and market_cap_col in available_target_cols and shares_outstanding_col in available_target_cols:
        # Detect correlated jumps using lazy expressions
        working_lf = working_lf.with_columns([
            (polars.col(market_cap_col) >= (polars.col(market_cap_col).shift(1) * 10)).alias('_mkt_cap_jump'),
            (polars.col(shares_outstanding_col) >= (polars.col(shares_outstanding_col).shift(1) * 10)).alias('_shares_jump'),
        ])

        # Mark rows with correlated jumps (both jump together)
        correlated_condition = polars.col('_mkt_cap_jump') & polars.col('_shares_jump')

        # For correlated jumps, null out the shares value to trigger forward fill
        working_lf = working_lf.with_columns(
            polars.when(correlated_condition)
            .then(None)
            .otherwise(polars.col(shares_outstanding_col))
            .forward_fill()
            .alias(shares_outstanding_col)
        )

        # Clean up temporary columns
        working_lf = working_lf.drop(['_mkt_cap_jump', '_shares_jump'])

    # 3. Cleaning Step - process remaining columns with standard lazy logic
    for col in available_target_cols:
        if col != shares_outstanding_col:  # Skip shares if already handled
            prev_val = polars.col(col).shift(1)
            col_condition = (polars.col(col) >= (prev_val * 10))

            working_lf = working_lf.with_columns(
                polars.when(col_condition)
                .then(None)
                .otherwise(polars.col(col))
                .forward_fill()
                .alias(col)
            )

    # Remove quarter column before returning
    working_lf = working_lf.drop("_quarter")

    if is_lazy:
        return working_lf, audit_log
    else:
        return working_lf.collect(), audit_log


def validate_financial_equivalencies(
    df: Union[polars.DataFrame, polars.LazyFrame],
    metadata: polars.LazyFrame,
    ticker: str,
    columns: list[str] = [""],
    date_col: str = "m_date",
    tolerance: float = 0.05,
    shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], dict]:
    """
    Validate and clean financial statement data by enforcing accounting identities.
    Processes data quarterly - evaluates once per quarter and logs with date ranges.

    Performs two types of validations:
    1. Hard Filters: Structural equations that are corrected via proportional scaling
       - Assets = Current Assets + Noncurrent Assets
       - Liabilities = Current Liabilities + Noncurrent Liabilities

    2. Soft Filters: Logical checks that flag warnings without modifying data
       - Stockholder Equity = Common Stock + APIC + Retained Earnings + Other Equity
       - Period End Cash = Cash and Cash Equivalents

    Args:
        df: Input DataFrame or LazyFrame with financial data
        ticker: Ticker symbol for logging purposes
        columns: Unused, kept for backward compatibility with parallel_process_tickers
        date_col: Name of the date column (default: 'm_date')
        tolerance: Tolerance for floating-point comparisons (default: 1.0)

    Returns:
        tuple containing:
            - Cleaned DataFrame/LazyFrame (same type as input)
            - Dictionary with keys 'hard_filter_errors' and 'soft_filter_warnings',
              each containing lists of error dictionaries with quarterly aggregation

    Note:
        Hard filter corrections use proportional scaling:
        - Factor = Total / (Current + Noncurrent)
        - NewComponent = OldComponent * Factor
        - Edge case: If components sum to 0 but total != 0, value goes to noncurrent
        Fundamental data is quarterly, so each quarter is evaluated once.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Add quarter column for grouping
    working_lf = _add_quarter_column(working_lf, date_col)

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())

    # Define required columns for hard and soft filters
    hard_filter_columns = {
        "assets": ["fbs_assets", "fbs_current_assets", "fbs_noncurrent_assets"],
        "liabilities": ["fbs_liabilities", "fbs_current_liabilities", "fbs_noncurrent_liabilities"]
    }

    soft_filter_columns = {
        "equity": [
            "fbs_stockholder_equity",
            "fbs_common_stock_value",
            "fbs_additional_paid_in_capital",
            "fbs_retained_earnings",
            "fbs_other_stockholder_equity"
        ],
        "cash": ["fcf_period_end_cash", "fbs_cash_and_cash_equivalents"],
        "balance_sheet": [
            "fbs_assets",
            "fbs_liabilities",
            "fbs_stockholder_equity",
            "fbs_noncontrolling_interest"
        ]
    }

    # Initialize error log structure
    error_log = {
        "hard_filter_errors": [],
        "soft_filter_warnings": []
    }

    # ==================== HARD FILTERS ====================
    # Track violation flags and correction expressions
    hard_violation_exprs = []
    hard_correction_exprs = []
    hard_violation_info = []
    columns_for_hard_logging = {date_col}

    # Process Assets identity
    if all(col in schema_cols for col in hard_filter_columns["assets"]):
        total_col = "fbs_assets"
        current_col = "fbs_current_assets"
        noncurrent_col = "fbs_noncurrent_assets"

        columns_for_hard_logging.update([total_col, current_col, noncurrent_col])

        # Calculate sum of components
        component_sum = polars.col(current_col) + polars.col(noncurrent_col)

        # Identify violations (absolute difference > tolerance)
        assets_violation = ((polars.col(total_col) - component_sum).abs() >
                            (polars.col(total_col).abs() * tolerance))

        # Calculate scaling factor
        scaling_factor = polars.when(component_sum != 0).then(
            polars.col(total_col) / component_sum
        ).otherwise(polars.lit(None))

        # Create correction expressions
        # Edge case: if sum is 0 but total is not, dump into noncurrent
        new_current = polars.when(assets_violation).then(
            polars.when(component_sum != 0).then(
                polars.col(current_col) * scaling_factor
            ).otherwise(polars.lit(0.0))
        ).otherwise(polars.col(current_col))

        new_noncurrent = polars.when(assets_violation).then(
            polars.when(component_sum != 0).then(
                polars.col(noncurrent_col) * scaling_factor
            ).otherwise(polars.col(total_col))  # Edge case: dump to noncurrent
        ).otherwise(polars.col(noncurrent_col))

        # Add to tracking lists
        violation_flag_name = "_viol_assets"
        hard_violation_exprs.append(assets_violation.alias(violation_flag_name))
        hard_correction_exprs.extend([
            new_current.alias(current_col),
            new_noncurrent.alias(noncurrent_col)
        ])
        hard_violation_info.append({
            "flag": violation_flag_name,
            "error_type": "assets_mismatch",
            "total_col": total_col,
            "current_col": current_col,
            "noncurrent_col": noncurrent_col
        })

    # Process Liabilities identity
    if all(col in schema_cols for col in hard_filter_columns["liabilities"]):
        total_col = "fbs_liabilities"
        current_col = "fbs_current_liabilities"
        noncurrent_col = "fbs_noncurrent_liabilities"

        columns_for_hard_logging.update([total_col, current_col, noncurrent_col])

        # Calculate sum of components
        component_sum = polars.col(current_col) + polars.col(noncurrent_col)

        # Identify violations (absolute difference > tolerance)
        liabilities_violation = ((polars.col(total_col) - component_sum).abs() >
                                 (polars.col(total_col)).abs() * tolerance)

        # Calculate scaling factor
        scaling_factor = polars.when(component_sum != 0).then(
            polars.col(total_col) / component_sum
        ).otherwise(polars.lit(None))

        # Create correction expressions
        new_current = polars.when(liabilities_violation).then(
            polars.when(component_sum != 0).then(
                polars.col(current_col) * scaling_factor
            ).otherwise(polars.lit(0.0))
        ).otherwise(polars.col(current_col))

        new_noncurrent = polars.when(liabilities_violation).then(
            polars.when(component_sum != 0).then(
                polars.col(noncurrent_col) * scaling_factor
            ).otherwise(polars.col(total_col))  # Edge case: dump to noncurrent
        ).otherwise(polars.col(noncurrent_col))

        # Add to tracking lists
        violation_flag_name = "_viol_liabilities"
        hard_violation_exprs.append(liabilities_violation.alias(violation_flag_name))
        hard_correction_exprs.extend([
            new_current.alias(current_col),
            new_noncurrent.alias(noncurrent_col)
        ])
        hard_violation_info.append({
            "flag": violation_flag_name,
            "error_type": "liabilities_mismatch",
            "total_col": total_col,
            "current_col": current_col,
            "noncurrent_col": noncurrent_col
        })

    # Log hard filter violations before correction - group by quarter
    if hard_violation_exprs:
        any_hard_violation = polars.any_horizontal(
            *[polars.col(info["flag"]) for info in hard_violation_info]
        )

        # Group by quarter and aggregate violations
        hard_error_query = (
            working_lf
            .select(list(columns_for_hard_logging) + ["_quarter"])
            .with_columns(hard_violation_exprs)
            .filter(any_hard_violation)
        )

        # For each violation type, group by quarter
        for info in hard_violation_info:
            total_col = info["total_col"]
            current_col = info["current_col"]
            noncurrent_col = info["noncurrent_col"]
            flag = info["flag"]

            quarter_errors = (
                hard_error_query
                .filter(polars.col(flag))
                .group_by("_quarter")
                .agg([
                    polars.col(date_col).min().alias("date_range_start"),
                    polars.col(date_col).max().alias("date_range_end"),
                    polars.col(total_col).first().alias("total"),
                    polars.col(current_col).first().alias("current"),
                    polars.col(noncurrent_col).first().alias("noncurrent"),
                    polars.len().alias("count")
                ])
                .sort("date_range_start")
                .limit(MAX_LOGS)
                .collect()
            )

            if quarter_errors.height > 0:
                error_rows = quarter_errors.to_dicts()

                for row in error_rows:
                    total_val = row.get("total")
                    current_val = row.get("current")
                    noncurrent_val = row.get("noncurrent")
                    component_sum_val = current_val + noncurrent_val

                    error_entry = {
                        "ticker": ticker,
                        "quarter": row.get("_quarter"),
                        "date_range_start": row.get("date_range_start"),
                        "date_range_end": row.get("date_range_end"),
                        "count": row.get("count"),
                        "error_type": info["error_type"],
                        "total": total_val,
                        "current": current_val,
                        "noncurrent": noncurrent_val,
                        "component_sum": component_sum_val,
                        "difference": total_val - component_sum_val
                    }

                    # Calculate corrected values
                    if component_sum_val != 0:
                        factor = total_val / component_sum_val
                        error_entry["corrected_current"] = current_val * factor
                        error_entry["corrected_noncurrent"] = noncurrent_val * factor
                        error_entry["correction_method"] = "proportional_scaling"
                    else:
                        error_entry["corrected_current"] = 0.0
                        error_entry["corrected_noncurrent"] = total_val
                        error_entry["correction_method"] = "residual_plug"

                    error_log["hard_filter_errors"].append(error_entry)

        # Apply hard filter corrections
        working_lf = working_lf.with_columns(hard_correction_exprs)

    # ==================== SOFT FILTERS ====================
    # Track violation flags for soft filters
    soft_violation_exprs = []
    soft_violation_info = []
    columns_for_soft_logging = {date_col}

    # Process Stockholder Equity identity
    if all(col in schema_cols for col in soft_filter_columns["equity"]):
        total_col = "fbs_stockholder_equity"
        component_cols = [
            "fbs_common_stock_value",
            "fbs_additional_paid_in_capital",
            "fbs_retained_earnings",
            "fbs_other_stockholder_equity"
        ]

        columns_for_soft_logging.update([total_col] + component_cols)

        # Calculate sum of components
        component_sum = sum(polars.col(col) for col in component_cols)

        # Identify violations
        equity_violation = ((polars.col(total_col) - component_sum).abs() >
                            (polars.col(total_col)).abs() * tolerance)

        violation_flag_name = "_warn_equity"
        soft_violation_exprs.append(equity_violation.alias(violation_flag_name))
        soft_violation_info.append({
            "flag": violation_flag_name,
            "error_type": "equity_mismatch",
            "total_col": total_col,
            "component_cols": component_cols
        })

    # Process Cash equivalency check
    if all(col in schema_cols for col in soft_filter_columns["cash"]):
        cash_col_1 = "fcf_period_end_cash"
        cash_col_2 = "fbs_cash_and_cash_equivalents"

        columns_for_soft_logging.update([cash_col_1, cash_col_2])

        # Identify violations
        cash_violation = ((polars.col(cash_col_1) - polars.col(cash_col_2)).abs() >
                          (polars.col(cash_col_1).abs() * tolerance))

        violation_flag_name = "_warn_cash"
        soft_violation_exprs.append(cash_violation.alias(violation_flag_name))
        soft_violation_info.append({
            "flag": violation_flag_name,
            "error_type": "cash_mismatch",
            "cash_col_1": cash_col_1,
            "cash_col_2": cash_col_2
        })

    # Log soft filter violations and create data_warning column
    data_warning_expr = polars.lit(False)

    # Process Fundamental Accounting Identity (Assets = Liabs + Equity + NCI)
    if all(col in schema_cols for col in soft_filter_columns["balance_sheet"]):
        assets_col = "fbs_assets"
        liabs_col = "fbs_liabilities"
        equity_col = "fbs_stockholder_equity"
        nci_col = "fbs_noncontrolling_interest"

        columns_for_soft_logging.update([assets_col, liabs_col, equity_col, nci_col])

        # Calculate Total Claims (Pasivo + Capital + Interés Minoritario)
        # Usamos fill_null(0) porque el NCI suele ser nulo en muchas empresas
        total_claims = (
                polars.col(liabs_col) +
                polars.col(equity_col) +
                polars.col(nci_col).fill_null(0.0)
        )

        # Identify violations (Dynamic Tolerance)
        bs_violation = (polars.col(assets_col) - total_claims).abs() > (polars.col(assets_col).abs() * tolerance)

        violation_flag_name = "_warn_balance_sheet"
        soft_violation_exprs.append(bs_violation.alias(violation_flag_name))
        soft_violation_info.append({
            "flag": violation_flag_name,
            "error_type": "accounting_equation_mismatch",  # A != L + E + NCI
            "total_col": assets_col,  # Usamos Activos como la verdad base
            # Pasamos las columnas componentes para el log detallado
            "component_cols": [liabs_col, equity_col, nci_col]
        })

    if soft_violation_exprs:
        # Create warning flag: True if any soft filter violation
        data_warning_expr = polars.any_horizontal(
            *[polars.col(info["flag"]) for info in soft_violation_info]
        )

        # Group by quarter and aggregate violations
        soft_error_query = (
            working_lf
            .select(list(columns_for_soft_logging) + ["_quarter"])
            .with_columns(soft_violation_exprs)
            .filter(data_warning_expr)
        )

        # For each violation type, group by quarter
        for info in soft_violation_info:
            flag = info["flag"]
            error_type = info["error_type"]

            # Build aggregation expressions based on error type
            if error_type == "equity_mismatch":
                total_col = info["total_col"]
                component_cols = info["component_cols"]

                agg_exprs = [
                    polars.col(date_col).min().alias("date_range_start"),
                    polars.col(date_col).max().alias("date_range_end"),
                    polars.col(total_col).first().alias("total"),
                    polars.len().alias("count")
                ] + [polars.col(col).first().alias(col) for col in component_cols]

            elif error_type == "accounting_equation_mismatch":
                total_col = info["total_col"]
                component_cols = info["component_cols"]

                agg_exprs = [
                    polars.col(date_col).min().alias("date_range_start"),
                    polars.col(date_col).max().alias("date_range_end"),
                    polars.col(total_col).first().alias("total"),
                    polars.len().alias("count")
                ] + [polars.col(col).first().alias(col) for col in component_cols]

            elif error_type == "cash_mismatch":
                cash_col_1 = info["cash_col_1"]
                cash_col_2 = info["cash_col_2"]

                agg_exprs = [
                    polars.col(date_col).min().alias("date_range_start"),
                    polars.col(date_col).max().alias("date_range_end"),
                    polars.col(cash_col_1).first().alias(cash_col_1),
                    polars.col(cash_col_2).first().alias(cash_col_2),
                    polars.len().alias("count")
                ]
            else:
                continue

            quarter_warnings = (
                soft_error_query
                .filter(polars.col(flag))
                .group_by("_quarter")
                .agg(agg_exprs)
                .sort("date_range_start")
                .limit(MAX_LOGS)
                .collect()
            )

            if quarter_warnings.height > 0:
                warning_rows = quarter_warnings.to_dicts()

                for row in warning_rows:
                    warning_entry = {
                        "ticker": ticker,
                        "quarter": row.get("_quarter"),
                        "date_range_start": row.get("date_range_start"),
                        "date_range_end": row.get("date_range_end"),
                        "count": row.get("count"),
                        "error_type": error_type
                    }

                    if error_type == "equity_mismatch":
                        total_col = info["total_col"]
                        component_cols = info["component_cols"]

                        total_val = row.get("total")
                        component_vals = {col: row.get(col) for col in component_cols}
                        component_sum_val = sum(component_vals.values())

                        warning_entry["total"] = total_val
                        warning_entry["components"] = component_vals
                        warning_entry["component_sum"] = component_sum_val
                        warning_entry["difference"] = total_val - component_sum_val

                    elif error_type == "accounting_equation_mismatch":
                        total_col = info["total_col"]
                        component_cols = info["component_cols"]

                        total_val = row.get("total")
                        # Handle nulls as 0 for the report
                        component_vals = {col: (row.get(col) or 0.0) for col in component_cols}
                        component_sum_val = sum(component_vals.values())

                        warning_entry["assets (total)"] = total_val
                        warning_entry["claims_sum"] = component_sum_val
                        warning_entry["components"] = component_vals
                        warning_entry["difference"] = total_val - component_sum_val

                    elif error_type == "cash_mismatch":
                        cash_col_1 = info["cash_col_1"]
                        cash_col_2 = info["cash_col_2"]

                        val_1 = row.get(cash_col_1)
                        val_2 = row.get(cash_col_2)

                        warning_entry[cash_col_1] = val_1
                        warning_entry[cash_col_2] = val_2
                        warning_entry["difference"] = val_1 - val_2

                    error_log["soft_filter_warnings"].append(warning_entry)

        # Add temporary violation flags to calculate data_warning
        working_lf = working_lf.with_columns(soft_violation_exprs)

    # Add data_warning column to the dataframe
    working_lf = working_lf.with_columns(
        data_warning_expr.alias("data_warning")
    )

    # Clean up temporary violation flag columns (only soft filter flags were added)
    if soft_violation_exprs:
        soft_flags_to_drop = [info["flag"] for info in soft_violation_info]
        working_lf = working_lf.drop(soft_flags_to_drop)

    # Remove quarter column before returning
    working_lf = working_lf.drop("_quarter")

    # Return in the same format as input
    result_df = working_lf if is_lazy else working_lf.collect()

    return (result_df, error_log)
