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
        if is_lazy:
            problem_rows = problem_query.collect()
        else:
            problem_rows = problem_query

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
        columns: list[str] = [""],
        date_col: str = 'm_date'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Ensures OHLC integrity for standard, split-adjusted, and dividend-adjusted columns.
    Works with both DataFrame (eager) and LazyFrame (lazy) execution.

    Logic:
      - Sets 'low' = min(open, high, low, close, vwap)
      - Sets 'high' = max(open, high, low, close, vwap)

    Groups processed:
      1. Standard: m_open, m_high, m_low, m_close
      2. Split Adj: m_open_split_adjusted, ...
      3. Div & Split Adj: m_open_dividend_and_split_adjusted, ...

    Args:
        df: Input polars DataFrame or LazyFrame.
        ticker: Stock ticker symbol for logging.
        columns: (unused in current implementation)
        date_col: Name of the date column for audit log.

    Returns:
        Tuple of (cleaned DataFrame/LazyFrame, audit_log list)

    Note:
        Assumes all OHLC columns are already correctly typed as Float64 via schema enforcement.
    """
    is_lazy = isinstance(df, polars.LazyFrame)

    # Define the column groups explicitly based on the subset provided
    groups = [
        # Group 1: Standard
        {
            "name": "standard",
            "open": "m_open",
            "high": "m_high",
            "low": "m_low",
            "close": "m_close",
            "vwap": "m_vwap"
        },
        # Group 2: Split Adjusted
        {
            "name": "split_adjusted",
            "open": "m_open_split_adjusted",
            "high": "m_high_split_adjusted",
            "low": "m_low_split_adjusted",
            "close": "m_close_split_adjusted",
            "vwap": "m_vwap_split_adjusted"
        },
        # Group 3: Dividend and Split Adjusted
        {
            "name": "dividend_and_split_adjusted",
            "open": "m_open_dividend_and_split_adjusted",
            "high": "m_high_dividend_and_split_adjusted",
            "low": "m_low_dividend_and_split_adjusted",
            "close": "m_close_dividend_and_split_adjusted",
            "vwap": "m_vwap_dividend_and_split_adjusted"
        }
    ]

    # ============================================================================
    # Step 1: Detect violations and build audit log
    # ============================================================================
    audit_expressions = []

    for g in groups:
        ohlc_cols = [g["open"], g["high"], g["low"], g["close"], g["vwap"]]

        # Filter to only columns that exist
        existing_ohlc_cols = [col for col in ohlc_cols if col in df.columns]

        if len(existing_ohlc_cols) >= 2:  # Need at least 2 columns for min/max
            # Calculate correct values
            audit_expressions.extend([
                polars.min_horizontal(existing_ohlc_cols).alias(f"_correct_low_{g['name']}"),
                polars.max_horizontal(existing_ohlc_cols).alias(f"_correct_high_{g['name']}")
            ])

    # Add these temporary columns for comparison
    df_with_audit = df.with_columns(audit_expressions)

    # Build violation detection expressions
    violation_conditions = []

    for g in groups:
        # Check if low or high need correction
        if g["low"] in df_with_audit.columns and f"_correct_low_{g['name']}" in df_with_audit.columns:
            low_violation = (
                    polars.col(g["low"]).is_not_null() &
                    (polars.col(g["low"]) != polars.col(f"_correct_low_{g['name']}"))
            )
        else:
            low_violation = polars.lit(False)

        if g["high"] in df_with_audit.columns and f"_correct_high_{g['name']}" in df_with_audit.columns:
            high_violation = (
                    polars.col(g["high"]).is_not_null() &
                    (polars.col(g["high"]) != polars.col(f"_correct_high_{g['name']}"))
            )
        else:
            high_violation = polars.lit(False)

        group_violation = low_violation | high_violation
        violation_conditions.append(group_violation)

    # Combine all violation conditions
    if violation_conditions:
        any_violation = violation_conditions[0]
        for cond in violation_conditions[1:]:
            any_violation = any_violation | cond
    else:
        any_violation = polars.lit(False)

    # Extract rows with violations
    select_cols = [date_col] if date_col in df_with_audit.columns else []

    for g in groups:
        group_cols = [
            g["open"], g["high"], g["low"], g["close"],
            f"_correct_low_{g['name']}", f"_correct_high_{g['name']}"
        ]
        select_cols.extend([col for col in group_cols if col in df_with_audit.columns])

    # Only proceed if we have columns to select
    if len(select_cols) > 0:
        problem_query = df_with_audit.filter(any_violation).select(select_cols)

        if is_lazy:
            problem_rows = problem_query.collect()
        else:
            problem_rows = problem_query

        # Build audit log
        audit_log = []
        if len(problem_rows) > 0:
            for row in problem_rows.iter_rows(named=True):
                date = row.get(date_col)

                for g in groups:
                    low_col = g["low"]
                    high_col = g["high"]
                    correct_low_col = f"_correct_low_{g['name']}"
                    correct_high_col = f"_correct_high_{g['name']}"

                    violations = []

                    # Check low violation
                    if (low_col in row and correct_low_col in row and
                            row[low_col] is not None and row[correct_low_col] is not None and
                            row[low_col] != row[correct_low_col]):
                        violations.append({
                            'column': low_col,
                            'original_value': row[low_col],
                            'corrected_value': row[correct_low_col],
                            'violation_type': 'low_too_high'
                        })

                    # Check high violation
                    if (high_col in row and correct_high_col in row and
                            row[high_col] is not None and row[correct_high_col] is not None and
                            row[high_col] != row[correct_high_col]):
                        violations.append({
                            'column': high_col,
                            'original_value': row[high_col],
                            'corrected_value': row[correct_high_col],
                            'violation_type': 'high_too_low'
                        })

                    # Add to audit log
                    for v in violations:
                        audit_log.append({
                            'ticker': ticker,
                            date_col: date,
                            'error_type': 'ohlc_integrity_violation',
                            'group': g['name'],
                            'column': v['column'],
                            'original_value': v['original_value'],
                            'corrected_value': v['corrected_value'],
                            'violation_type': v['violation_type']
                        })

            logging.info(f"Found {len(audit_log)} OHLC integrity violations for ticker {ticker}")
        else:
            audit_log = []
    else:
        audit_log = []

    # ============================================================================
    # Step 2: Apply corrections
    # ============================================================================
    correction_expressions = []

    for g in groups:
        ohlc_cols = [g["open"], g["high"], g["low"], g["close"], g["vwap"]]

        # Filter to only existing columns
        existing_ohlc_cols = [col for col in ohlc_cols if col in df.columns]

        if len(existing_ohlc_cols) >= 2:
            # Create expression to overwrite the 'low' column
            if g["low"] in df.columns:
                correction_expressions.append(
                    polars.min_horizontal(existing_ohlc_cols).alias(g["low"])
                )

            # Create expression to overwrite the 'high' column
            if g["high"] in df.columns:
                correction_expressions.append(
                    polars.max_horizontal(existing_ohlc_cols).alias(g["high"])
                )

    # Apply all corrections at once
    if correction_expressions:
        df_corrected = df.with_columns(correction_expressions)
    else:
        df_corrected = df

    return df_corrected, audit_log


def fill_negatives_market(
        df: Union[polars.DataFrame, polars.LazyFrame],
        ticker: str,
        columns: list[str],
        date_col: str = 'm_date'
) -> tuple[Union[polars.DataFrame, polars.LazyFrame], list[dict]]:
    """
    Detects negative values in specified columns and imputes them using cubic spline
    interpolation based ONLY on previous valid data (no forward-looking bias).

    This implements the temporal imputation methodology from Section 5.1 of the
    Financial Data Error Detection Framework: "Spline Interpolation: For missing
    m_close prices, a cubic spline fits a smooth curve through the data points.
    Unlike linear interpolation, which creates artificial kinks, splines preserve
    the smoothness of the price path."

    Works with both DataFrame (eager) and LazyFrame (lazy) execution.

    Logic:
      1. Sort by date to ensure temporal ordering
      2. For each column, identify negative values
      3. For each negative value:
         - Collect all PREVIOUS valid (non-negative) data points
         - If ≥3 previous points: Use cubic spline interpolation
         - If 2 previous points: Use linear interpolation
         - If 1 previous point: Use last observation carried forward (LOCF)
         - If 0 previous points: Use next valid observation carried backward (NOCB)
      4. Log all corrections in audit trail

    Constraint Violations Detected:
      - Negative prices (violates Constraint 2 from Section 2.1.1:
        "Prices cannot be negative in equity markets")
      - Negative volumes (impossible in market microstructure)
      - Negative fundamental values where sign convention expects positive

    Args:
        df: Input polars DataFrame or LazyFrame.
        ticker: Stock ticker symbol for logging.
        columns: List of column names to check for negative values.
                 Example: ['m_open', 'm_close', 'm_volume', 'fbs_assets']
        date_col: Name of the date column for temporal ordering.

    Returns:
        Tuple of (cleaned DataFrame/LazyFrame, audit_log list)

    Raises:
        ValueError: If date_col not in dataframe or if columns list is empty.

    Note:
        Assumes columns are already correctly typed as Float64 via schema enforcement.
    """

    # ============================================================================
    # Input Validation
    # ============================================================================
    if not columns or len(columns) == 0:
        raise ValueError("columns parameter cannot be empty")

    is_lazy = isinstance(df, polars.LazyFrame)

    # For lazy frames, collect first to work with actual data
    # This is necessary because we need to access data for spline interpolation
    if is_lazy:
        df_working = df.collect()
        was_lazy = True
    else:
        df_working = df
        was_lazy = False

    if date_col not in df_working.columns:
        raise ValueError(f"date_col '{date_col}' not found in dataframe columns")

    # Filter columns to only those that exist in the dataframe
    existing_columns = [col for col in columns if col in df_working.columns]

    if len(existing_columns) == 0:
        logging.warning(f"None of the specified columns exist in dataframe for ticker {ticker}")
        return (df.lazy() if was_lazy else df), []

    # ============================================================================
    # Step 1: Sort by Date (CRITICAL for forward-looking bias prevention)
    # ============================================================================
    df_working = df_working.sort(date_col)

    # ============================================================================
    # Step 2: Detect Negative Values and Build Audit Log
    # ============================================================================
    audit_log = []
    correction_map = {}  # Store corrections: {(row_idx, col): corrected_value}

    # Convert to pandas for easier row-wise operations (polars doesn't have good row indexing)
    df_pandas = df_working.to_pandas()

    for col in existing_columns:
        # Find indices where values are negative
        negative_mask = df_pandas[col] < 0
        negative_indices = df_pandas[negative_mask].index.tolist()

        if len(negative_indices) == 0:
            continue  # No negative values in this column

        logging.info(f"Found {len(negative_indices)} negative values in column '{col}' for ticker {ticker}")

        # ========================================================================
        # Step 3: Impute Each Negative Value Using Backward-Looking Spline
        # ========================================================================
        for neg_idx in negative_indices:
            original_value = df_pandas.loc[neg_idx, col]
            date_value = df_pandas.loc[neg_idx, date_col]

            # Collect ALL PREVIOUS valid (non-negative, non-null) data points
            previous_mask = (
                    (df_pandas.index < neg_idx) &  # Only previous rows
                    (df_pandas[col] >= 0) &  # Non-negative
                    (df_pandas[col].notna())  # Non-null
            )
            previous_data = df_pandas[previous_mask]

            corrected_value = None
            imputation_method = None

            # Choose imputation method based on available previous data
            if len(previous_data) >= 3:
                # ================================================================
                # Method 1: Cubic Spline Interpolation (Preferred)
                # ================================================================
                # Extract previous dates and values
                prev_indices = previous_data.index.tolist()
                prev_values = previous_data[col].values

                # Create numeric x-axis (use index positions for interpolation)
                x_prev = numpy.array(prev_indices)
                y_prev = numpy.array(prev_values)

                try:
                    # Fit cubic spline through previous valid points
                    # bc_type='natural' ensures smooth second derivative at boundaries
                    cs = CubicSpline(x_prev, y_prev, bc_type='natural')

                    # Interpolate at the negative value's position
                    corrected_value = float(cs(neg_idx))

                    # Ensure corrected value is non-negative (spline can overshoot)
                    if corrected_value < 0:
                        corrected_value = float(y_prev[-1])  # Fall back to LOCF
                        imputation_method = 'cubic_spline_with_locf_fallback'
                    else:
                        imputation_method = 'cubic_spline'

                except Exception as e:
                    # Spline fitting can fail with collinear points
                    logging.warning(f"Cubic spline failed for {ticker} col={col} idx={neg_idx}: {e}")
                    corrected_value = float(previous_data[col].iloc[-1])
                    imputation_method = 'locf_after_spline_failure'

            elif len(previous_data) == 2:
                # ================================================================
                # Method 2: Linear Interpolation
                # ================================================================
                prev_indices = previous_data.index.tolist()
                prev_values = previous_data[col].values

                # Linear extrapolation from last two points
                x1, x2 = prev_indices[-2], prev_indices[-1]
                y1, y2 = prev_values[-2], prev_values[-1]

                # Linear equation: y = y1 + (y2-y1)/(x2-x1) * (x-x1)
                slope = (y2 - y1) / (x2 - x1)
                corrected_value = float(y1 + slope * (neg_idx - x1))

                # Ensure non-negative
                if corrected_value < 0:
                    corrected_value = float(y2)  # Use last observation
                    imputation_method = 'linear_with_locf_fallback'
                else:
                    imputation_method = 'linear_interpolation'

            elif len(previous_data) == 1:
                # ================================================================
                # Method 3: Last Observation Carried Forward (LOCF)
                # ================================================================
                corrected_value = float(previous_data[col].iloc[-1])
                imputation_method = 'locf'

            else:
                # ================================================================
                # Method 4: Next Observation Carried Backward (NOCB)
                # ================================================================
                # No previous valid data - must look forward (unavoidable)
                next_mask = (
                        (df_pandas.index > neg_idx) &
                        (df_pandas[col] >= 0) &
                        (df_pandas[col].notna())
                )
                next_data = df_pandas[next_mask]

                if len(next_data) > 0:
                    corrected_value = float(next_data[col].iloc[0])
                    imputation_method = 'nocb_no_previous_data'
                else:
                    # No valid data at all - use 0 or column median
                    valid_mask = (df_pandas[col] >= 0) & (df_pandas[col].notna())
                    if valid_mask.any():
                        corrected_value = float(df_pandas.loc[valid_mask, col].median())
                        imputation_method = 'column_median_fallback'
                    else:
                        corrected_value = 0.0
                        imputation_method = 'zero_fallback_no_valid_data'

            # Store correction
            correction_map[(neg_idx, col)] = corrected_value

            # Add to audit log
            audit_log.append({
                'ticker': ticker,
                date_col: date_value,
                'row_index': int(neg_idx),
                'error_type': 'negative_value_violation',
                'column': col,
                'original_value': float(original_value),
                'corrected_value': corrected_value,
                'imputation_method': imputation_method,
                'previous_valid_points': len(previous_data) if previous_data is not None else 0
            })

    # ============================================================================
    # Step 4: Apply All Corrections
    # ============================================================================
    if len(correction_map) > 0:
        # Apply corrections to pandas dataframe
        for (row_idx, col), corrected_val in correction_map.items():
            df_pandas.loc[row_idx, col] = corrected_val

        # Convert back to polars
        df_corrected = polars.from_pandas(df_pandas)

        logging.info(f"Corrected {len(audit_log)} negative values for ticker {ticker}")
    else:
        df_corrected = df_working
        logging.info(f"No negative values found for ticker {ticker}")

    # ============================================================================
    # Step 5: Return in Original Format (Lazy or Eager)
    # ============================================================================
    if was_lazy:
        return df_corrected.lazy(), audit_log
    else:
        return df_corrected, audit_log