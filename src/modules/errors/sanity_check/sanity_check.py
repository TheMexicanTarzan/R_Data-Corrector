import polars
import numpy
from typing import Union, Literal
import logging
from scipy.interpolate import CubicSpline


def sort_dates(
        df: Union[polars.DataFrame, polars.LazyFrame],
        ticker: str,
        columns: list[str],
        date_col: str = "m_date",
        dedupe_strategy: Literal["earliest", "latest", "all"] = "all",
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Sort a polars DataFrame or LazyFrame by date columns with a defined hierarchy.

    Sorting hierarchy (highest to lowest priority):
    1. f_filing_date
    2. f_accepted_date
    3. date_col (default: m_date)

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
        "f_filing_date",
        "f_accepted_date",
        date_col,
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

    # Get schema to check column types
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

    # Add original index and cast date columns, then collect ONCE
    df_collected = (
        df.lazy()
        .with_row_index("_original_idx")
        .with_columns(cast_expressions)
        .collect()
    )

    # Split into valid and null rows using boolean mask (single pass)
    valid_mask = df_collected[primary_sort_col].is_not_null()
    valid_rows = df_collected.filter(valid_mask)
    null_rows = df_collected.filter(~valid_mask)

    if valid_rows.height > 0:
        # Get original positions as Series (avoid Python list conversion)
        valid_original_positions = valid_rows["_original_idx"].clone()

        # Sort valid rows by date columns
        sorted_valid = valid_rows.sort(by=temp_sort_cols)

        # Assign new indices: sorted rows get original valid positions in order
        sorted_valid = sorted_valid.with_columns(
            valid_original_positions.alias("_new_idx")
        )
    else:
        sorted_valid = valid_rows.with_columns(
            polars.col("_original_idx").alias("_new_idx")
        )

    # Null rows keep their original positions
    null_rows = null_rows.with_columns(
        polars.col("_original_idx").alias("_new_idx")
    )

    # Combine and sort by new index to restore proper order
    combined = polars.concat([sorted_valid, null_rows]).sort("_new_idx")

    # Detect mismatches (only for valid rows since null rows don't move)
    mismatches = combined.filter(
        (polars.col("_original_idx") != polars.col("_new_idx")) &
        polars.col(primary_sort_col).is_not_null()
    ).select(["_original_idx", "_new_idx"] + sort_columns)

    if mismatches.height > 0:
        mismatch_dicts = mismatches.to_dicts()
        for row in mismatch_dicts:
            row["ticker"] = ticker
            row["error_type"] = "order_mismatch"
            row["original_position"] = row.pop("_original_idx")
            row["sorted_position"] = row.pop("_new_idx")
        logs.extend(mismatch_dicts)

    # Remove temporary columns
    result_df = combined.drop(["_original_idx", "_new_idx"] + temp_sort_cols)

    if dedupe_strategy != "all" and date_col in columns:
        declaration_col = "f_filing_date"
        if declaration_col in columns:
            pre_dedupe_count = result_df.height

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
                    .collect()
                )
            elif dedupe_strategy == "latest":
                result_df = (
                    result_df.lazy()
                    .with_columns(dedupe_cast_exprs)
                    .sort(by=temp_dedupe_cols, descending=[False, True])
                    .drop(temp_dedupe_cols)
                    .unique(subset=[date_col], keep="first", maintain_order=True)
                    .collect()
                )

            # Final sort by the main sort columns (preserving null positions)
            result_df = (
                result_df
                .with_row_index("_orig_idx")
                .with_columns(cast_expressions)
            )

            valid_mask_dedupe = result_df[primary_sort_col].is_not_null()
            valid_dedupe = result_df.filter(valid_mask_dedupe)
            null_dedupe = result_df.filter(~valid_mask_dedupe)

            if valid_dedupe.height > 0:
                valid_positions = valid_dedupe["_orig_idx"].clone()
                sorted_dedupe = valid_dedupe.sort(by=temp_sort_cols)
                sorted_dedupe = sorted_dedupe.with_columns(
                    valid_positions.alias("_new_idx")
                )
            else:
                sorted_dedupe = valid_dedupe.with_columns(
                    polars.col("_orig_idx").alias("_new_idx")
                )

            null_dedupe = null_dedupe.with_columns(
                polars.col("_orig_idx").alias("_new_idx")
            )

            result_df = (
                polars.concat([sorted_dedupe, null_dedupe])
                .sort("_new_idx")
                .drop(["_orig_idx", "_new_idx"] + temp_sort_cols)
            )

            removed_count = pre_dedupe_count - result_df.height
            if removed_count > 0:
                logs.append(
                    {
                        "ticker": ticker,
                        "error_type": "duplicates_removed",
                        "strategy": dedupe_strategy,
                        "duplicates_removed": removed_count,
                    }
                )

    # Convert to lazy if input was lazy
    if is_lazy:
        result_df = result_df.lazy()

    return result_df, logs


def fill_negatives_fundamentals(
        df: Union[polars.DataFrame, polars.LazyFrame],
        columns: list[str],
        ticker: str,
        date_col: str = 'm_date'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], dict[str, list[dict]]]:
    """
    Replaces negative values in specified columns with the last non-negative value.
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
                {'ticker': 'AAPL', 'date': '...', 'value': -5},
                ...
            ]
        }

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    audit_log = {}

    # 1. Audit Step - requires collecting problem rows
    for col in columns:
        # Build query for problem rows
        problem_query = df.filter(polars.col(col) < 0).select([date_col, col])

        # Collect only the problem rows (not the entire dataset)
        problem_rows = problem_query.collect()

        # Check if there are any problem rows
        if len(problem_rows) > 0:
            # Convert to dicts and inject the ticker string into every entry
            entries = problem_rows.to_dicts()
            for entry in entries:
                entry['ticker'] = ticker
            audit_log[col] = entries

    # 2. Cleaning Step - works lazily or eagerly
    cleaned_df = df.with_columns(
        (
            polars.when(polars.col(col) >= 0)
            .then(polars.col(col))
            .otherwise(None)
            .forward_fill()
            .alias(col)
        )
        for col in columns
    )

    return cleaned_df, audit_log


def fill_negatives_market(
        df: Union[polars.DataFrame, polars.LazyFrame],
        ticker: str,
        columns: list[str],
        date_col: str = 'm_date'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detect and correct negative values in market data using backward-looking cubic spline.

    Uses only previous data points to avoid look-forward bias, making it suitable
    for backtesting applications.

    Args:
        df: Input DataFrame or LazyFrame containing market data
        ticker: Ticker symbol for logging/tracking purposes
        columns: List of column names to check and correct for negative values
        date_col: Name of the date column (default: 'm_date')

    Returns:
        tuple containing:
            - Corrected DataFrame/LazyFrame (same type as input)
            - List of dictionaries documenting each correction made
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    if is_lazy:
        working_df = df.collect()
    else:
        working_df = df.clone()

    working_df = working_df.sort(date_col)

    corrections = []

    for col in columns:
        if col not in working_df.columns:
            continue

        values = working_df[col].to_numpy().astype(numpy.float64)
        dates = working_df[date_col].to_list()

        # Track original null positions to preserve them
        null_mask = ~numpy.isfinite(values)

        # Find negatives: must be finite AND less than zero
        negative_mask = numpy.isfinite(values) & (values < 0)
        negative_indices = numpy.where(negative_mask)[0]

        if len(negative_indices) == 0:
            continue

        for idx in negative_indices:
            original_value = float(values[idx])

            prev_valid_indices = []
            prev_valid_values = []
            for i in range(idx - 1, -1, -1):
                if values[i] >= 0 and numpy.isfinite(values[i]):
                    prev_valid_indices.append(i)
                    prev_valid_values.append(values[i])
                if len(prev_valid_indices) >= 4:
                    break

            if len(prev_valid_indices) == 0:
                corrections.append({
                    'ticker': ticker,
                    'column': col,
                    'date': dates[idx],
                    'original_value': original_value,
                    'corrected_value': None,
                    'method': 'skipped_no_previous_valid'
                })
                continue

            if len(prev_valid_indices) < 3:
                corrected_value = prev_valid_values[0]
                method = 'last_valid_value'
            else:
                prev_valid_indices = prev_valid_indices[::-1]
                prev_valid_values = prev_valid_values[::-1]

                spline = CubicSpline(
                    prev_valid_indices,
                    prev_valid_values,
                    extrapolate=True
                )
                corrected_value = float(spline(idx))
                method = 'cubic_spline_backward'

                if corrected_value < 0 or not numpy.isfinite(corrected_value):
                    corrected_value = prev_valid_values[-1]
                    method = 'last_valid_value_fallback'

            values[idx] = corrected_value

            corrections.append({
                'ticker': ticker,
                'column': col,
                'date': dates[idx],
                'original_value': original_value,
                'corrected_value': corrected_value,
                'method': method
            })

        # Restore original null positions
        values[null_mask] = numpy.nan

        working_df = working_df.with_columns(
            polars.Series(name=col, values=values)
        )

    if is_lazy:
        return working_df.lazy(), corrections
    else:
        return working_df, corrections


def zero_wipeout(
        df: Union[polars.DataFrame, polars.LazyFrame],
        columns: list[str],
        ticker: str,
        date_col: str = 'm_date'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Identifies rows where share columns are 0 AND 'm_volume' > 0.
    These 0 values are replaced via forward fill.
    Works with both DataFrame (eager) and LazyFrame (lazy) execution.

    Args:
        df: Input polars DataFrame or LazyFrame.
        columns: List of column names to check for zero wipeout.
        ticker: String representing the ticker symbol.
        date_col: Name of the column containing dates.

    Returns:
        tuple: (Cleaned DataFrame/LazyFrame, Audit List)

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
    """
    import polars as pl

    is_lazy = isinstance(df, pl.LazyFrame)
    target_cols = columns

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

    # 1. Audit Step - collect problem rows
    problem_query = df.filter(final_condition).select(
        [date_col] + target_cols + ["m_volume"]
    )

    # Collect only the problem rows
    if is_lazy:
        problem_rows = problem_query.collect()
    else:
        problem_rows = problem_query

    audit_log = []
    if len(problem_rows) > 0:
        entries = problem_rows.to_dicts()
        for entry in entries:
            entry['ticker'] = ticker
        audit_log = entries

    # 2. Cleaning Step - process each column separately
    # For each target column, replace 0s (when volume > 0) with None, then forward fill
    for col in target_cols:
        # Condition for this specific column
        col_condition = (pl.col(col) == 0) & (pl.col("m_volume") > 0)

        df = df.with_columns(
            pl.when(col_condition)
            .then(None)  # Replace problematic 0s with null
            .otherwise(pl.col(col))  # Keep everything else
            .forward_fill()  # Fill nulls with previous values
            .alias(col)
        )

    return df, audit_log


def mkt_cap_scale_error(
        df: Union[polars.DataFrame, polars.LazyFrame],
        ticker: str,
        columns: list[str] = [""],
        date_col: str = 'm_date',
        market_cap_col: str = 'c_market_cap',
        shares_outstanding_col: str = 'fis_weighted_average_diluted_shares_outstanding'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Identifies and corrects rows where share columns jump by 10x or more
    compared to the previous day. If both market cap and shares outstanding
    jump together, forward fills the entire span of the shares outstanding error.

    Args:
        df: Input polars DataFrame or LazyFrame.
        ticker: String representing the ticker symbol.
        columns: List of column names to check for scale errors.
        date_col: Name of the column containing dates.
        market_cap_col: Name of the market cap column.
        shares_outstanding_col: Name of the shares outstanding column.

    Returns:
        tuple: (Cleaned DataFrame/LazyFrame, Audit List)

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    target_cols = columns

    # Check if both market cap and shares outstanding columns exist
    has_market_cap = market_cap_col in df.columns
    has_shares = shares_outstanding_col in df.columns

    # Build combined condition for audit
    conditions = []
    for col in target_cols:
        if col in df.columns:
            prev_shares = polars.col(col).shift(1)
            col_condition = (polars.col(col) >= (prev_shares * 10))
            conditions.append(col_condition)

    if not conditions:
        return df, []

    combined_condition = conditions[0]
    for cond in conditions[1:]:
        combined_condition = combined_condition | cond

    # 1. Audit Step
    select_cols = [date_col] + [col for col in target_cols if col in df.columns]

    problem_query = df.filter(combined_condition).select(select_cols)

    if is_lazy:
        problem_rows = problem_query.collect()
    else:
        problem_rows = problem_query

    audit_log = []
    if len(problem_rows) > 0:
        entries = problem_rows.to_dicts()
        for entry in entries:
            entry['ticker'] = ticker
            entry['error_type'] = "scale_10x_jump"
        audit_log = entries
        logging.info(f"Found {len(audit_log)} scale errors (10x jump) for ticker {ticker}")

    # 2. Check for correlated market cap and shares outstanding jumps
    if has_market_cap and has_shares and market_cap_col in target_cols and shares_outstanding_col in target_cols:
        # Collect to work with the data
        working_df = df.collect() if is_lazy else df.clone()

        # Find jumps in both market cap and shares outstanding
        working_df = working_df.with_columns([
            (polars.col(market_cap_col) >= (polars.col(market_cap_col).shift(1) * 10)).alias('_mkt_cap_jump'),
            (polars.col(shares_outstanding_col) >= (polars.col(shares_outstanding_col).shift(1) * 10)).alias(
                '_shares_jump'),
            polars.col(shares_outstanding_col).shift(1).alias('_prev_shares')
        ])

        # Find rows where BOTH jump
        correlated_jumps = working_df.filter(
            polars.col('_mkt_cap_jump') & polars.col('_shares_jump')
        )

        if len(correlated_jumps) > 0:
            # For each correlated jump, find the span of elevated shares
            jump_dates = correlated_jumps.select(date_col).to_series().to_list()

            for jump_date in jump_dates:
                # Get the last good shares value before the jump
                pre_jump = working_df.filter(polars.col(date_col) < jump_date).tail(1)

                if len(pre_jump) > 0:
                    last_good_shares = pre_jump.select(shares_outstanding_col).item()

                    # Get the jumped value
                    jumped_row = working_df.filter(polars.col(date_col) == jump_date)
                    if len(jumped_row) > 0:
                        jumped_shares = jumped_row.select(shares_outstanding_col).item()

                        # Find the span where shares remain at the elevated level
                        # (within 20% of the jumped value, indicating same scale error)
                        working_df = working_df.with_columns([
                            polars.when(
                                (polars.col(date_col) >= jump_date) &
                                (polars.col(shares_outstanding_col) >= (jumped_shares * 0.8)) &
                                (polars.col(shares_outstanding_col) <= (jumped_shares * 1.2))
                            )
                            .then(polars.lit(True))
                            .otherwise(polars.lit(False))
                            .alias('_in_error_span')
                        ])

        # Clean up temporary columns
        working_df = working_df.drop(['_mkt_cap_jump', '_shares_jump', '_prev_shares', '_in_error_span'], strict=False)

        # Convert back to lazy if needed
        df = working_df.lazy() if is_lazy else working_df

    # 3. Cleaning Step - process remaining columns with standard row-by-row logic
    for col in target_cols:
        if col in df.columns and col != shares_outstanding_col:  # Skip shares if already handled
            prev_shares = polars.col(col).shift(1)
            col_condition = (polars.col(col) >= (prev_shares * 10))

            df = df.with_columns(
                polars.when(col_condition)
                .then(None)
                .otherwise(polars.col(col))
                .forward_fill()
                .alias(col)
            )

    return df, audit_log


def ohlc_integrity(
    df: Union[polars.DataFrame, polars.LazyFrame],
    ticker: str,
    columns: list[str] = [""],  # for backward compatibility, useless but do not eliminate
    date_col: str = "m_date",
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Validate and resolve OHLC data integrity issues:
    - High must be >= max(Open, Close, Low) -> resolved by setting High = max
    - Low must be <= min(Open, Close, High) -> resolved by setting Low = min
    - VWAP must be within [Low, High] -> resolved by set VWAP = (O+H+L+C)/4

    Checks are performed on three column groups:
    - Raw OHLC
    - Split adjusted OHLC
    - Dividend and split adjusted OHLC

    Args:
        df: Polars DataFrame or LazyFrame containing OHLC data
        ticker: Ticker symbol for error reporting
        columns: Unused, kept for backward compatibility
        date_col: Name of the date column

    Returns:
        Tuple of (corrected dataframe/lazyframe, list of resolution dictionaries)
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence without collecting
    schema_cols = set(working_lf.collect_schema().names())

    # Define column groups to validate
    column_groups = [
        {
            "name": "raw",
            "open": "m_open",
            "high": "m_high",
            "low": "m_low",
            "close": "m_close",
            "vwap": "m_vwap",
        },
        {
            "name": "split_adjusted",
            "open": "m_open_split_adjusted",
            "high": "m_high_split_adjusted",
            "low": "m_low_split_adjusted",
            "close": "m_close_split_adjusted",
            "vwap": "m_vwap_split_adjusted",
        },
        {
            "name": "dividend_and_split_adjusted",
            "open": "m_open_dividend_and_split_adjusted",
            "high": "m_high_dividend_and_split_adjusted",
            "low": "m_low_dividend_and_split_adjusted",
            "close": "m_close_dividend_and_split_adjusted",
            "vwap": "m_vwap_dividend_and_split_adjusted",
        },
    ]

    # Build expressions
    violation_exprs = []
    correction_exprs = []
    violation_info = []
    columns_for_logging = {date_col}

    for group in column_groups:
        group_name = group["name"]
        open_col = group["open"]
        high_col = group["high"]
        low_col = group["low"]
        close_col = group["close"]
        vwap_col = group["vwap"]

        # Check if required OHLC columns exist
        required_cols = [open_col, high_col, low_col, close_col]
        if not all(col in schema_cols for col in required_cols):
            continue

        has_vwap = vwap_col in schema_cols

        # Track columns needed for logging
        columns_for_logging.update(required_cols)
        if has_vwap:
            columns_for_logging.add(vwap_col)

        # Precompute expressions (Polars will optimize common subexpressions)
        ohlc_max = polars.max_horizontal(
            polars.col(open_col),
            polars.col(high_col),
            polars.col(low_col),
            polars.col(close_col),
        )
        ohlc_min = polars.min_horizontal(
            polars.col(open_col),
            polars.col(high_col),
            polars.col(low_col),
            polars.col(close_col),
        )

        # Check 1: High >= max(Open, Close, Low)
        high_viol_name = f"_viol_{group_name}_high"
        high_viol_expr = polars.col(high_col) < ohlc_max
        violation_exprs.append(high_viol_expr.alias(high_viol_name))
        correction_exprs.append(
            polars.when(high_viol_expr)
            .then(ohlc_max)
            .otherwise(polars.col(high_col))
            .alias(high_col)
        )
        violation_info.append(
            {
                "col": high_viol_name,
                "error_type": "high_not_maximum",
                "group_name": group_name,
                "group": group,
                "has_vwap": has_vwap,
            }
        )

        # Check 2: Low <= min(Open, Close, High)
        low_viol_name = f"_viol_{group_name}_low"
        low_viol_expr = polars.col(low_col) > ohlc_min
        violation_exprs.append(low_viol_expr.alias(low_viol_name))
        correction_exprs.append(
            polars.when(low_viol_expr)
            .then(ohlc_min)
            .otherwise(polars.col(low_col))
            .alias(low_col)
        )
        violation_info.append(
            {
                "col": low_viol_name,
                "error_type": "low_not_minimum",
                "group_name": group_name,
                "group": group,
                "has_vwap": has_vwap,
            }
        )

        # Check 3: VWAP within [Low, High]
        if has_vwap:
            ohlc_centroid = (
                polars.col(open_col)
                + polars.col(high_col)
                + polars.col(low_col)
                + polars.col(close_col)
            ) / 4.0

            vwap_viol_name = f"_viol_{group_name}_vwap"
            vwap_viol_expr = (polars.col(vwap_col) < polars.col(low_col)) | (
                polars.col(vwap_col) > polars.col(high_col)
            )
            violation_exprs.append(vwap_viol_expr.alias(vwap_viol_name))
            correction_exprs.append(
                polars.when(vwap_viol_expr)
                .then(ohlc_centroid)
                .otherwise(polars.col(vwap_col))
                .alias(vwap_col)
            )
            violation_info.append(
                {
                    "col": vwap_viol_name,
                    "error_type": "vwap_outside_range",
                    "group_name": group_name,
                    "group": group,
                    "has_vwap": has_vwap,
                }
            )

    # If no checks possible, return early
    if not violation_exprs:
        return (df, [])

    # Build logging query: select only needed columns, add flags, filter violations
    violation_col_names = [info["col"] for info in violation_info]
    any_violation = polars.any_horizontal(*[polars.col(c) for c in violation_col_names])

    error_rows_df = (
        working_lf
        .select(list(columns_for_logging))
        .with_columns(violation_exprs)
        .filter(any_violation)
        .collect()
    )

    # Build resolution logs
    resolutions = []
    if not error_rows_df.is_empty():
        error_rows = error_rows_df.to_dicts()

        for row in error_rows:
            for info in violation_info:
                if not row.get(info["col"]):
                    continue

                group = info["group"]
                group_name = info["group_name"]
                error_type = info["error_type"]
                has_vwap = info["has_vwap"]

                open_col = group["open"]
                high_col = group["high"]
                low_col = group["low"]
                close_col = group["close"]
                vwap_col = group["vwap"]

                open_val = row.get(open_col)
                high_val = row.get(high_col)
                low_val = row.get(low_col)
                close_val = row.get(close_col)
                vwap_val = row.get(vwap_col) if has_vwap else None

                ohlc_vals = [open_val, high_val, low_val, close_val]
                ohlc_max_val = max(ohlc_vals)
                ohlc_min_val = min(ohlc_vals)

                resolution_entry = {
                    "ticker": ticker,
                    "date": row.get(date_col),
                    "error_type": error_type,
                    "column_group": group_name,
                    "open": open_val,
                    "high": high_val,
                    "low": low_val,
                    "close": close_val,
                    "vwap": vwap_val,
                }

                if error_type == "high_not_maximum":
                    resolution_entry["old_high"] = high_val
                    resolution_entry["new_high"] = ohlc_max_val
                    resolution_entry["message"] = (
                        f"High corrected from {high_val} to {ohlc_max_val}"
                    )
                elif error_type == "low_not_minimum":
                    resolution_entry["old_low"] = low_val
                    resolution_entry["new_low"] = ohlc_min_val
                    resolution_entry["message"] = (
                        f"Low corrected from {low_val} to {ohlc_min_val}"
                    )
                elif error_type == "vwap_outside_range":
                    ohlc_centroid_val = sum(ohlc_vals) / 4.0
                    resolution_entry["old_vwap"] = vwap_val
                    resolution_entry["new_vwap"] = ohlc_centroid_val
                    resolution_entry["message"] = (
                        f"VWAP corrected from {vwap_val} to {ohlc_centroid_val} (OHLC centroid)"
                    )

                resolutions.append(resolution_entry)

    # Apply corrections (separate query, no violation flags in output)
    corrected_lf = working_lf.with_columns(correction_exprs)

    return (corrected_lf, resolutions)


def validate_financial_equivalencies(
    df: Union[polars.DataFrame, polars.LazyFrame],
    ticker: str,
    columns: list[str] = [""],
    date_col: str = "m_date",
    tolerance: float = 0.05
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], dict]:
    """
    Validate and clean financial statement data by enforcing accounting identities.

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
              each containing lists of error dictionaries

    Note:
        Hard filter corrections use proportional scaling:
        - Factor = Total / (Current + Noncurrent)
        - NewComponent = OldComponent * Factor
        - Edge case: If components sum to 0 but total != 0, value goes to noncurrent
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

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

    # Log hard filter violations before correction
    if hard_violation_exprs:
        any_hard_violation = polars.any_horizontal(
            *[polars.col(info["flag"]) for info in hard_violation_info]
        )

        hard_error_rows_df = (
            working_lf
            .select(list(columns_for_hard_logging))
            .with_columns(hard_violation_exprs)
            .filter(any_hard_violation)
            .collect()
        )

        if not hard_error_rows_df.is_empty():
            error_rows = hard_error_rows_df.to_dicts()

            for row in error_rows:
                for info in hard_violation_info:
                    if not row.get(info["flag"]):
                        continue

                    total_col = info["total_col"]
                    current_col = info["current_col"]
                    noncurrent_col = info["noncurrent_col"]

                    total_val = row.get(total_col)
                    current_val = row.get(current_col)
                    noncurrent_val = row.get(noncurrent_col)
                    component_sum_val = current_val + noncurrent_val

                    error_entry = {
                        "ticker": ticker,
                        "date": row.get(date_col),
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

        soft_error_rows_df = (
            working_lf
            .select(list(columns_for_soft_logging))
            .with_columns(soft_violation_exprs)
            .filter(data_warning_expr)
            .collect()
        )

        if not soft_error_rows_df.is_empty():
            error_rows = soft_error_rows_df.to_dicts()

            for row in error_rows:
                for info in soft_violation_info:
                    if not row.get(info["flag"]):
                        continue

                    warning_entry = {
                        "ticker": ticker,
                        "date": row.get(date_col),
                        "error_type": info["error_type"]
                    }

                    if info["error_type"] == "equity_mismatch":
                        total_col = info["total_col"]
                        component_cols = info["component_cols"]

                        total_val = row.get(total_col)
                        component_vals = {col: row.get(col) for col in component_cols}
                        component_sum_val = sum(component_vals.values())

                        warning_entry["total"] = total_val
                        warning_entry["components"] = component_vals
                        warning_entry["component_sum"] = component_sum_val
                        warning_entry["difference"] = total_val - component_sum_val

                    elif info["error_type"] == "accounting_equation_mismatch":
                        total_col = info["total_col"]
                        component_cols = info["component_cols"]

                        total_val = row.get(total_col)
                        # Handle nulls as 0 for the report
                        component_vals = {col: (row.get(col) or 0.0) for col in component_cols}
                        component_sum_val = sum(component_vals.values())

                        warning_entry["assets (total)"] = total_val
                        warning_entry["claims_sum"] = component_sum_val
                        warning_entry["components"] = component_vals
                        warning_entry["difference"] = total_val - component_sum_val

                    elif info["error_type"] == "cash_mismatch":
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

    # Return in the same format as input
    result_df = working_lf if is_lazy else working_lf.collect()

    return (result_df, error_log)

