import polars
import numpy
from typing import Union
import logging
from scipy.interpolate import CubicSpline


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


import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline


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
    - VWAP must be within [Low, High] -> resolved by setting VWAP = (O+H+L+C)/4

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