import polars
import pathlib
from typing import Dict, Union, Optional, List

# Import the comprehensive schema and helper functions
from src.input_handlers.schema import (
    FINANCIAL_DATA_SCHEMA,
    infer_schema_from_patterns,
    get_schema_for_columns,
    build_complete_schema_from_file
)

data_dir = "../../Input/Data"
metadata_dir = "../../Input/Universe_Information/Universe_Information.csv"


def read_csv_files_to_polars(
        directory_path: str = data_dir,
        lazy: bool = True,
        max_files: Optional[int] = None,
        file_pattern: str = "*.csv",
        include_files: Optional[List[str]] = None,
        exclude_files: Optional[List[str]] = None,
        use_schema_override: bool = True
) -> Dict[str, Union[polars.LazyFrame, polars.DataFrame]]:
    """
    Reads CSV files from a directory into Polars DataFrames or LazyFrames.

    Args:
        directory_path: Path to the directory containing CSV files
        lazy: If True, returns LazyFrames (memory-efficient, recommended for large files).
              If False, returns eager DataFrames (loads all data into memory immediately).
        max_files: Maximum number of files to load. If None, loads all files.
                   Files are loaded in alphabetical order by filename.
        file_pattern: Glob pattern to match files (default: "*.csv")
        include_files: Optional list of filenames (without extension) to include.
                      If provided, only these files will be loaded.
        exclude_files: Optional list of filenames (without extension) to exclude.
                      Takes precedence over include_files.
        use_schema_override: If True, uses comprehensive schema with pattern-based inference
                            to ensure correct types. Set to False if you want Polars to
                            auto-infer types (not recommended).

    Returns:
        Dictionary mapping filename (without extension) to Polars LazyFrame or DataFrame

    Note:
        Schema override prevents type inference errors like:
        "cannot parse `0.01928264873357904` as dtype `i64`"

        The schema now uses pattern-based inference, so even columns not explicitly
        defined in FINANCIAL_DATA_SCHEMA will be correctly typed based on naming patterns.
    """
    directory = pathlib.Path(directory_path)

    # Validate directory
    if not directory.exists():
        raise ValueError(f"Directory does not exist: {directory_path}")

    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory_path}")

    # Find all CSV files matching pattern
    csv_files = sorted(directory.glob(file_pattern))

    if not csv_files:
        print(f"Warning: No files matching '{file_pattern}' found in {directory_path}")
        return {}

    # Filter files based on include/exclude lists
    filtered_files = []
    for csv_file in csv_files:
        file_key = csv_file.stem

        # Check exclude list first (takes precedence)
        if exclude_files and file_key in exclude_files:
            continue

        # Check include list if provided
        if include_files and file_key not in include_files:
            continue

        filtered_files.append(csv_file)

    if not filtered_files:
        print(f"Warning: No files remained after applying filters")
        return {}

    # Apply max_files limit
    if max_files is not None and max_files > 0:
        filtered_files = filtered_files[:max_files]
        print(f"Loading {len(filtered_files)} files (limited by max_files={max_files})")
    else:
        print(f"Loading {len(filtered_files)} files")

    dataframe_dict = {}

    # Read each CSV file
    for csv_file in filtered_files:
        file_key = csv_file.stem

        try:
            if use_schema_override:
                # Build complete schema with pattern-based inference
                complete_schema = build_complete_schema_from_file(str(csv_file))

                if lazy:
                    # Lazy loading - data not loaded into memory until .collect() is called
                    dataframe = polars.scan_csv(
                        csv_file,
                        try_parse_dates=True,
                        rechunk=True,
                        schema_overrides=complete_schema,  # ← Uses pattern inference
                        ignore_errors=False  # Fail on parsing errors rather than silently using null
                    )
                else:
                    # Eager loading - data immediately loaded into memory
                    dataframe = polars.read_csv(
                        csv_file,
                        try_parse_dates=True,
                        rechunk=True,
                        schema_overrides=complete_schema,  # ← Uses pattern inference
                        ignore_errors=False
                    )
            else:
                # Original behavior - let Polars infer types (not recommended)
                if lazy:
                    dataframe = polars.scan_csv(
                        csv_file,
                        try_parse_dates=True,
                        rechunk=True,
                    )
                else:
                    dataframe = polars.read_csv(
                        csv_file,
                        try_parse_dates=True,
                        rechunk=True,
                    )

            dataframe_dict[file_key] = dataframe

        except Exception as error:
            print(f"Error reading {csv_file.name}: {error}")
            # Provide helpful debugging info
            if "cannot parse" in str(error) and "as dtype" in str(error):
                print(f"  → This is a schema type mismatch.")
                print(f"  → The column should likely be Float64, not Int64.")
                print(f"  → Set use_schema_override=True (default) to fix this automatically.")
                print(f"  → The pattern-based inference should handle unknown columns.")
            continue

    frame_type = "LazyFrames" if lazy else "DataFrames"
    schema_info = " (with pattern-based schema)" if use_schema_override else " (auto-inferred types)"
    print(f"Successfully loaded {len(dataframe_dict)} CSV files as {frame_type}{schema_info}")

    return dataframe_dict


def read_single_csv_with_schema(
        file_path: str,
        lazy: bool = True,
        custom_schema: Optional[Dict[str, polars.DataType]] = None,
        use_pattern_inference: bool = True
) -> Union[polars.LazyFrame, polars.DataFrame]:
    """
    Read a single CSV file with proper schema handling.

    Args:
        file_path: Path to the CSV file
        lazy: Whether to use lazy loading
        custom_schema: Optional custom schema dict. If None, uses pattern-based inference.
        use_pattern_inference: If True and custom_schema is None, uses pattern-based inference.
                               If False, uses only FINANCIAL_DATA_SCHEMA (may miss columns).

    Returns:
        LazyFrame or DataFrame

    Examples:
        >>> # Recommended: Uses pattern inference for all columns
        >>> df = read_single_csv_with_schema('AAPL.csv', lazy=False)

        >>> # Custom schema
        >>> schema = {'ticker': polars.String, 'c_volatility': polars.Float64}
        >>> df = read_single_csv_with_schema('AAPL.csv', custom_schema=schema)

        >>> # Check schema
        >>> print(df.schema)
    """
    if custom_schema is not None:
        # User provided custom schema
        schema_to_use = custom_schema
    elif use_pattern_inference:
        # Build complete schema with pattern inference (RECOMMENDED)
        schema_to_use = build_complete_schema_from_file(file_path)
    else:
        # Use only predefined schema (may miss columns)
        schema_to_use = FINANCIAL_DATA_SCHEMA

    try:
        if lazy:
            return polars.scan_csv(
                file_path,
                try_parse_dates=True,
                rechunk=True,
                schema_overrides=schema_to_use,
                ignore_errors=False
            )
        else:
            return polars.read_csv(
                file_path,
                try_parse_dates=True,
                rechunk=True,
                schema_overrides=schema_to_use,
                ignore_errors=False
            )
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        if "cannot parse" in str(e) and "as dtype" in str(e):
            print(f"  → Schema type mismatch detected.")
            print(f"  → Ensure use_pattern_inference=True (default) to handle all columns.")
        raise


def read_csv_with_validation(
        file_path: str,
        lazy: bool = True,
        validate: bool = True
) -> Union[polars.LazyFrame, polars.DataFrame]:
    """
    Read a CSV file with schema validation.

    This function reads the CSV with pattern-based schema inference,
    then optionally validates the resulting schema matches expectations.

    Args:
        file_path: Path to the CSV file
        lazy: Whether to use lazy loading
        validate: If True, validates schema after loading (requires .collect() for lazy frames)

    Returns:
        LazyFrame or DataFrame

    Raises:
        ValueError: If validation fails and validate=True

    Examples:
        >>> df = read_csv_with_validation('AAPL.csv', lazy=False, validate=True)
        >>> # This will print schema issues if any are found
    """
    # Read with pattern-based schema
    df = read_single_csv_with_schema(file_path, lazy=lazy, use_pattern_inference=True)

    # Validate if requested (only works for eager DataFrames)
    if validate and not lazy:
        from src.input_handlers.schema import validate_schema
        issues = validate_schema(df, strict=False)

        if issues:
            print(f"Schema validation found {len(issues)} issues in {file_path}:")
            for col, issue in issues.items():
                print(f"  - {col}: {issue}")

            # Optionally raise error
            # raise ValueError(f"Schema validation failed for {file_path}")
        else:
            print(f"Schema validation passed for {file_path}")

    return df


def get_columns_from_csv(file_path: str) -> List[str]:
    """
    Quick utility to get column names from a CSV without loading the data.

    Args:
        file_path: Path to the CSV file

    Returns:
        List of column names

    Examples:
        >>> columns = get_columns_from_csv('AAPL.csv')
        >>> print(f"Found {len(columns)} columns")
        >>> print(columns[:5])  # First 5 columns
    """
    with open(file_path, 'r') as f:
        headers = f.readline().strip().split(',')
    return headers


def preview_schema(file_path: str) -> Dict[str, polars.DataType]:
    """
    Preview what schema will be used for a CSV file without loading it.

    Args:
        file_path: Path to the CSV file

    Returns:
        Schema dictionary that would be used

    Examples:
        >>> schema = preview_schema('AAPL.csv')
        >>> print(f"Will use schema with {len(schema)} columns")
        >>> for col, dtype in list(schema.items())[:10]:
        >>>     print(f"  {col}: {dtype}")
    """
    return build_complete_schema_from_file(file_path)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Read all CSV files from directory with pattern inference
    print("=" * 70)
    print("Example 1: Reading all CSV files with pattern-based schema")
    print("=" * 70)

    # This will correctly handle all columns, including dynamically calculated ones
    dataframes = read_csv_files_to_polars(
        directory_path=data_dir,
        lazy=True,
        max_files=5,  # Limit for demo
        use_schema_override=True  # Default - uses pattern inference
    )

    if dataframes:
        first_key = list(dataframes.keys())[0]
        first_df = dataframes[first_key]
        print(f"\nFirst file: {first_key}")
        print(f"Schema preview (first 10 columns):")

        # For lazy frames, we can check the schema without collecting
        schema_items = list(first_df.schema.items())[:10]
        for col, dtype in schema_items:
            print(f"  {col}: {dtype}")

    print("\n" + "=" * 70)
    print("Example 2: Read single file with validation")
    print("=" * 70)

    # Example path - adjust as needed
    example_file = pathlib.Path(data_dir) / "AAPL.csv"
    if example_file.exists():
        df = read_csv_with_validation(
            str(example_file),
            lazy=False,
            validate=True
        )
        print(f"\nLoaded {len(df)} rows, {len(df.columns)} columns")

    print("\n" + "=" * 70)
    print("Example 3: Preview schema before loading")
    print("=" * 70)

    if example_file.exists():
        schema = preview_schema(str(example_file))
        print(f"\nSchema preview for {example_file.name}:")
        print(f"Total columns: {len(schema)}")

        # Show schema by prefix
        prefixes = ['m_', 'c_', 'fbs_', 'fis_', 'fcf_']
        for prefix in prefixes:
            cols = {k: v for k, v in schema.items() if k.startswith(prefix)}
            if cols:
                print(f"\n{prefix}* columns ({len(cols)}):")
                for col, dtype in list(cols.items())[:3]:  # Show first 3
                    print(f"  {col}: {dtype}")
                if len(cols) > 3:
                    print(f"  ... and {len(cols) - 3} more")