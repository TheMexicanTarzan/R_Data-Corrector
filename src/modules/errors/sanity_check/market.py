import polars
import numpy
from typing import Union
from scipy.interpolate import CubicSpline


def fill_negatives_market(
        df: Union[polars.DataFrame, polars.LazyFrame],
        metadata: polars.LazyFrame,
        ticker: str,
        columns: list[str],
        date_col: str = 'm_date',
        shared_data: dict = None  # Unused - for interface consistency
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

    # MEMORY FIX: First check if there are ANY negatives using lazy query
    # Only collect if we actually need to do corrections
    working_lf = df.lazy()

    # Build a condition to check for any negatives in any column
    negative_check_exprs = []
    available_cols = []
    schema = working_lf.collect_schema()

    for col in columns:
        if col in schema.names():
            available_cols.append(col)
            negative_check_exprs.append(polars.col(col) < 0)

    if not available_cols:
        # No columns to process
        return (df, [])

    # Check if any negatives exist (lightweight query)
    any_negative_expr = negative_check_exprs[0]
    for expr in negative_check_exprs[1:]:
        any_negative_expr = any_negative_expr | expr

    has_negatives = working_lf.filter(any_negative_expr).limit(1).collect().height > 0

    if not has_negatives:
        # No negatives, return as-is without collecting
        return (df, [])

    # MEMORY FIX: Only collect the columns we need, not the entire dataframe
    needed_cols = [date_col] + available_cols
    working_df = working_lf.select(needed_cols).sort(date_col).collect()

    corrections = []
    MAX_CORRECTIONS_LOG = 50  # Limit log entries per column

    for col in available_cols:
        values = working_df[col].to_numpy().astype(numpy.float64)
        dates = working_df[date_col].to_list()

        # Track original null positions to preserve them
        null_mask = ~numpy.isfinite(values)

        # Find negatives: must be finite AND less than zero
        negative_mask = numpy.isfinite(values) & (values < 0)
        negative_indices = numpy.where(negative_mask)[0]

        if len(negative_indices) == 0:
            continue

        col_corrections_logged = 0
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
                if col_corrections_logged < MAX_CORRECTIONS_LOG:
                    corrections.append({
                        'ticker': ticker,
                        'column': col,
                        'date': dates[idx],
                        'original_value': original_value,
                        'corrected_value': None,
                        'method': 'skipped_no_previous_valid'
                    })
                    col_corrections_logged += 1
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

            # MEMORY FIX: Limit log entries
            if col_corrections_logged < MAX_CORRECTIONS_LOG:
                corrections.append({
                    'ticker': ticker,
                    'column': col,
                    'date': dates[idx],
                    'original_value': original_value,
                    'corrected_value': corrected_value,
                    'method': method
                })
                col_corrections_logged += 1

        # Restore original null positions
        values[null_mask] = numpy.nan

        working_df = working_df.with_columns(
            polars.Series(name=col, values=values)
        )

    # MEMORY FIX: Join corrected columns back to original lazy frame instead of returning full collected df
    # Create a lazy frame with just the corrected columns and join by index
    corrected_cols_df = working_df.select(available_cols)

    # Add row index to both for joining
    result_lf = (
        df.lazy()
        .sort(date_col)
        .with_row_index("_join_idx")
        .drop(available_cols)  # Remove original columns
        .join(
            corrected_cols_df.lazy().with_row_index("_join_idx"),
            on="_join_idx",
            how="left"
        )
        .drop("_join_idx")
    )

    if is_lazy:
        return result_lf, corrections
    else:
        return result_lf.collect(), corrections


def ohlc_integrity(
    df: Union[polars.DataFrame, polars.LazyFrame],
    metadata: polars.LazyFrame,
    ticker: str,
    columns: list[str] = [""],  # for backward compatibility, useless but do not eliminate
    date_col: str = "m_date",
    shared_data: dict = None  # Unused - for interface consistency
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



def validate_market_split_consistency(
    df: Union[polars.DataFrame, polars.LazyFrame],
    metadata: polars.LazyFrame,
    ticker: str,
    columns: list[str] = [""],
    date_col: str = "m_date",
    tolerance: float = 0.01,
    shared_data: dict = None  # Unused - for interface consistency
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Validate that the relationship between raw market data and split-adjusted market data
    is mathematically consistent with the explicit split events declared in the dataset.

    The function computes an expected cumulative adjustment factor (K_expected) from
    explicit split events and compares it against the implied factor (K_implied) derived
    from raw/adjusted column pairs. Rows where the implied factor deviates beyond the
    tolerance are flagged and corrected.

    Logic:
        1. Calculate Daily Event Factor: factor = s_split_date_denominator / s_split_date_numerator
           - If numerator/denominator are 0 or Null, assume factor = 1.0 (no split)
        2. Calculate Cumulative K (K_expected): cumulative product of daily factors
        3. For price columns: K_implied = adjusted / raw
        4. For volume: K_implied = raw / adjusted (inverse relationship)
        5. Validate: |K_implied - K_expected| <= tolerance * |K_expected|

    Explicit Column Pairs Validated:
        - Split event: s_split_date_numerator, s_split_date_denominator
        - Price pairs: m_open/m_open_split_adjusted, m_high/m_high_split_adjusted,
                       m_low/m_low_split_adjusted, m_close/m_close_split_adjusted,
                       m_vwap/m_vwap_split_adjusted
        - Volume pair: m_volume/m_volume_split_adjusted

    Args:
        df: Input Polars LazyFrame or DataFrame containing market data
        ticker: Ticker symbol for logging purposes
        columns: List of column names to analyze (used to identify available pairs)
        date_col: Name of the date column (default: 'm_date')
        tolerance: Acceptable margin of error for validation (default: 1% = 0.01)

    Returns:
        tuple containing:
            - Corrected DataFrame/LazyFrame (same type as input)
            - List of dictionaries documenting corrections and skipped pairs

    Note:
        Corrections recalculate adjusted values using K_expected:
        - For prices: corrected_adjusted = raw * K_expected
        - For volume: corrected_adjusted = raw / K_expected
    """
    is_lazy = isinstance(df, polars.LazyFrame)
    working_lf = df if is_lazy else df.lazy()

    # Get schema to check column existence
    schema_cols = set(working_lf.collect_schema().names())

    # Initialize log as a flat list
    logs = []

    # Define the explicit split event columns
    split_numerator_col = "s_split_date_numerator"
    split_denominator_col = "s_split_date_denominator"

    # Define explicit price metric pairs to validate
    all_price_pairs = [
        ("m_open", "m_open_split_adjusted"),
        ("m_high", "m_high_split_adjusted"),
        ("m_low", "m_low_split_adjusted"),
        ("m_close", "m_close_split_adjusted"),
        ("m_vwap", "m_vwap_split_adjusted"),
    ]

    # Define volume pair (inverse relationship)
    volume_pair = ("m_volume", "m_volume_split_adjusted")

    # Check for split columns first - if missing, we cannot validate
    has_split_cols = (
        split_numerator_col in schema_cols and
        split_denominator_col in schema_cols
    )

    if not has_split_cols:
        logs.append({
            "ticker": ticker,
            "error_type": "skipped_validation",
            "reason": "missing_split_columns",
            "missing": [col for col in [split_numerator_col, split_denominator_col] if col not in schema_cols]
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # Filter pairs based on columns parameter - only validate columns that are requested
    columns_set = set(columns) if columns and columns != [""] else schema_cols

    # Check which pairs are available and log skipped pairs
    available_price_pairs = []
    for raw_col, adj_col in all_price_pairs:
        # Check if pair is requested
        is_requested = (raw_col in columns_set or adj_col in columns_set or columns == [""])
        if not is_requested:
            continue

        # Check if both columns exist in schema
        if raw_col not in schema_cols or adj_col not in schema_cols:
            missing = []
            if raw_col not in schema_cols:
                missing.append(raw_col)
            if adj_col not in schema_cols:
                missing.append(adj_col)
            logs.append({
                "ticker": ticker,
                "error_type": "skipped_pair",
                "pair_type": "price",
                "raw_column": raw_col,
                "adjusted_column": adj_col,
                "reason": "missing_columns",
                "missing": missing
            })
            continue

        available_price_pairs.append((raw_col, adj_col))

    # Check volume pair
    has_volume_pair = False
    raw_vol, adj_vol = volume_pair
    is_volume_requested = (raw_vol in columns_set or adj_vol in columns_set or columns == [""])

    if is_volume_requested:
        if raw_vol not in schema_cols or adj_vol not in schema_cols:
            missing = []
            if raw_vol not in schema_cols:
                missing.append(raw_vol)
            if adj_vol not in schema_cols:
                missing.append(adj_vol)
            logs.append({
                "ticker": ticker,
                "error_type": "skipped_pair",
                "pair_type": "volume",
                "raw_column": raw_vol,
                "adjusted_column": adj_vol,
                "reason": "missing_columns",
                "missing": missing
            })
        else:
            has_volume_pair = True

    # If no pairs to validate, return early
    if not available_price_pairs and not has_volume_pair:
        logs.append({
            "ticker": ticker,
            "error_type": "skipped_validation",
            "reason": "no_valid_pairs_available"
        })
        result_df = working_lf if is_lazy else working_lf.collect()
        return (result_df, logs)

    # ==================== STEP 1: Calculate Daily Event Factor ====================
    # factor = denominator / numerator
    # If numerator or denominator is 0 or Null, factor = 1.0 (no split event)
    daily_factor_expr = (
        polars.when(
            (polars.col(split_numerator_col).is_null()) |
            (polars.col(split_denominator_col).is_null()) |
            (polars.col(split_numerator_col) == 0) |
            (polars.col(split_denominator_col) == 0)
        )
        .then(polars.lit(1.0))
        .otherwise(
            polars.col(split_denominator_col) / polars.col(split_numerator_col)
        )
    ).alias("_daily_factor")

    # ==================== STEP 2: Calculate Cumulative K (K_expected) ====================
    # K_expected = cumulative product of daily factors
    # We use log-sum-exp approach for numerical stability: K = exp(sum(log(factor)))
    cumulative_k_expr = (
        polars.col("_daily_factor").log().cum_sum().exp()
    ).alias("_k_expected")

    # Add the computed columns
    working_lf = working_lf.with_columns([daily_factor_expr])
    working_lf = working_lf.with_columns([cumulative_k_expr])

    # ==================== STEP 3 & 4: Validate Price and Volume Columns ====================
    # Build validation expressions for all pairs
    # Note: We need separate lists because violation flags depend on k_implied columns
    k_implied_exprs = []
    violation_flag_exprs = []
    correction_exprs = []
    violation_info = []
    columns_for_logging = {date_col, "_k_expected", "_daily_factor"}

    # Process price pairs
    for raw_col, adj_col in available_price_pairs:
        columns_for_logging.update([raw_col, adj_col])

        # K_implied = adjusted / raw (for prices)
        k_implied_name = f"_k_implied_{raw_col}"
        k_implied_expr = (
            polars.when(
                (polars.col(raw_col).is_null()) |
                (polars.col(raw_col) == 0)
            )
            .then(polars.lit(None))
            .otherwise(polars.col(adj_col) / polars.col(raw_col))
        ).alias(k_implied_name)

        # Violation check: |K_implied - K_expected| > tolerance * |K_expected|
        # Also check that both raw and adjusted are valid (not null/zero)
        violation_name = f"_viol_{raw_col}"
        violation_expr = (
            polars.col(raw_col).is_not_null() &
            (polars.col(raw_col) != 0) &
            polars.col(adj_col).is_not_null() &
            (
                (polars.col(k_implied_name) - polars.col("_k_expected")).abs() >
                (polars.col("_k_expected").abs() * tolerance)
            )
        ).alias(violation_name)

        # Correction: recalculate adjusted = raw * K_expected
        corrected_adj_expr = (
            polars.when(polars.col(violation_name))
            .then(polars.col(raw_col) * polars.col("_k_expected"))
            .otherwise(polars.col(adj_col))
        ).alias(adj_col)

        k_implied_exprs.append(k_implied_expr)
        violation_flag_exprs.append(violation_expr)
        correction_exprs.append(corrected_adj_expr)

        violation_info.append({
            "flag": violation_name,
            "k_implied_col": k_implied_name,
            "error_type": "price_split_mismatch",
            "raw_col": raw_col,
            "adj_col": adj_col,
            "relationship": "price"
        })

    # Process volume pair (inverse relationship)
    if has_volume_pair:
        raw_col, adj_col = volume_pair
        columns_for_logging.update([raw_col, adj_col])

        # K_implied = raw / adjusted (for volume - inverse relationship)
        k_implied_name = f"_k_implied_{raw_col}"
        k_implied_expr = (
            polars.when(
                (polars.col(adj_col).is_null()) |
                (polars.col(adj_col) == 0)
            )
            .then(polars.lit(None))
            .otherwise(polars.col(raw_col) / polars.col(adj_col))
        ).alias(k_implied_name)

        # Violation check
        violation_name = f"_viol_{raw_col}"
        violation_expr = (
            polars.col(adj_col).is_not_null() &
            (polars.col(adj_col) != 0) &
            polars.col(raw_col).is_not_null() &
            (
                (polars.col(k_implied_name) - polars.col("_k_expected")).abs() >
                (polars.col("_k_expected").abs() * tolerance)
            )
        ).alias(violation_name)

        # Correction: recalculate adjusted = raw / K_expected
        corrected_adj_expr = (
            polars.when(polars.col(violation_name))
            .then(
                polars.when(polars.col("_k_expected") != 0)
                .then(polars.col(raw_col) / polars.col("_k_expected"))
                .otherwise(polars.col(adj_col))
            )
            .otherwise(polars.col(adj_col))
        ).alias(adj_col)

        k_implied_exprs.append(k_implied_expr)
        violation_flag_exprs.append(violation_expr)
        correction_exprs.append(corrected_adj_expr)

        violation_info.append({
            "flag": violation_name,
            "k_implied_col": k_implied_name,
            "error_type": "volume_split_mismatch",
            "raw_col": raw_col,
            "adj_col": adj_col,
            "relationship": "volume"
        })

    # Apply k_implied expressions first, then violation flags (which depend on k_implied)
    working_lf = working_lf.with_columns(k_implied_exprs)
    working_lf = working_lf.with_columns(violation_flag_exprs)

    # ==================== STEP 5: Log Violations and Apply Corrections ====================
    if violation_info:
        # Create any_violation expression
        violation_flags = [polars.col(info["flag"]) for info in violation_info]
        any_violation = polars.any_horizontal(*violation_flags)

        # Collect error rows for logging
        logging_cols = list(columns_for_logging)
        k_implied_cols = [info["k_implied_col"] for info in violation_info]
        flag_cols = [info["flag"] for info in violation_info]

        error_rows_df = (
            working_lf
            .select(logging_cols + k_implied_cols + flag_cols)
            .filter(any_violation)
            .collect()
        )

        if not error_rows_df.is_empty():
            error_rows = error_rows_df.to_dicts()

            for row in error_rows:
                for info in violation_info:
                    if not row.get(info["flag"]):
                        continue

                    raw_col = info["raw_col"]
                    adj_col = info["adj_col"]
                    k_implied_col = info["k_implied_col"]

                    raw_val = row.get(raw_col)
                    adj_val = row.get(adj_col)
                    k_implied = row.get(k_implied_col)
                    k_expected = row.get("_k_expected")
                    daily_factor = row.get("_daily_factor")

                    # Calculate corrected value
                    if info["relationship"] == "price":
                        corrected_val = raw_val * k_expected if raw_val is not None else None
                    else:  # volume
                        corrected_val = raw_val / k_expected if (raw_val is not None and k_expected != 0) else None

                    logs.append({
                        "ticker": ticker,
                        "date": row.get(date_col),
                        "error_type": info["error_type"],
                        "raw_column": raw_col,
                        "adjusted_column": adj_col,
                        "raw_value": raw_val,
                        "original_adjusted_value": adj_val,
                        "corrected_adjusted_value": corrected_val,
                        "k_expected": k_expected,
                        "k_implied": k_implied,
                        "daily_split_factor": daily_factor,
                        "deviation": abs(k_implied - k_expected) if (k_implied is not None and k_expected is not None) else None,
                        "tolerance": tolerance,
                        "correction_method": "recalculated_from_k_expected"
                    })

        # Apply corrections
        working_lf = working_lf.with_columns(correction_exprs)

    # ==================== CLEANUP: Remove Temporary Columns ====================
    temp_cols_to_drop = ["_daily_factor", "_k_expected"]
    temp_cols_to_drop.extend([info["k_implied_col"] for info in violation_info])
    temp_cols_to_drop.extend([info["flag"] for info in violation_info])

    working_lf = working_lf.drop(temp_cols_to_drop)

    # Return in the same format as input
    result_df = working_lf if is_lazy else working_lf.collect()

    return (result_df, logs)
