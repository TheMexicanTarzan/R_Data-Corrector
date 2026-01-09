"""
Output handler for saving corrected fundamental data to CSV or Parquet files.
"""
import polars as pl
from pathlib import Path
from typing import Union, Tuple, Dict, Optional, Literal
import logging

logger = logging.getLogger(__name__)


def save_corrected_data(
    clean_data_dict: Dict[str, Tuple[Union[pl.LazyFrame, pl.DataFrame], pl.LazyFrame]],
    output_directory: Union[str, Path],
    file_format: Literal["csv", "parquet"] = "csv",
    create_directory: bool = True,
    overwrite: bool = True
) -> Dict[str, Path]:
    """
    Save corrected fundamental data as CSV or Parquet files.

    This function takes the dictionary of clean LazyFrames from data_corrector.py
    and saves them to disk in the specified format.

    Parameters
    ----------
    clean_data_dict : Dict[str, Tuple[Union[pl.LazyFrame, pl.DataFrame], pl.LazyFrame]]
        Dictionary where:
        - Keys: filenames (e.g., "AAPL.csv")
        - Values: Tuple of (ticker_data_lf, metadata_lf)
            - ticker_data_lf: LazyFrame or DataFrame containing the corrected ticker data
            - metadata_lf: LazyFrame containing metadata (not saved, only ticker data is saved)

    output_directory : Union[str, Path]
        Directory path where corrected data files will be saved.
        Will be created if it doesn't exist and create_directory=True.

    file_format : Literal["csv", "parquet"], default="csv"
        Output file format. Options:
        - "csv": Save as CSV files (human-readable, larger file size)
        - "parquet": Save as Parquet files (compressed, smaller file size, faster I/O)

    create_directory : bool, default=True
        If True, creates the output directory if it doesn't exist.
        If False and directory doesn't exist, raises FileNotFoundError.

    overwrite : bool, default=True
        If True, overwrites existing files with the same name.
        If False, skips files that already exist.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping original filenames to their saved file paths.
        Keys: original filenames (e.g., "AAPL.csv")
        Values: Path objects to the saved files

    Raises
    ------
    FileNotFoundError
        If output_directory doesn't exist and create_directory=False.
    ValueError
        If file_format is not "csv" or "parquet".
    TypeError
        If clean_data_dict values are not tuples or don't contain LazyFrame/DataFrame.

    Examples
    --------
    >>> # Save as CSV files
    >>> saved_files = save_corrected_data(
    ...     clean_data_dict=clean_lfs,
    ...     output_directory="../Output/CorrectedData",
    ...     file_format="csv"
    ... )

    >>> # Save as Parquet files
    >>> saved_files = save_corrected_data(
    ...     clean_data_dict=clean_lfs,
    ...     output_directory="../Output/CorrectedData",
    ...     file_format="parquet"
    ... )

    Notes
    -----
    - LazyFrames are collected (materialized) before saving
    - CSV files use comma delimiter and include headers
    - Parquet files use Snappy compression by default
    - Progress is logged for every 100 files processed
    """
    # Validate file format
    if file_format not in ["csv", "parquet"]:
        raise ValueError(f"file_format must be 'csv' or 'parquet', got '{file_format}'")

    # Convert output_directory to Path object
    output_dir = Path(output_directory)

    # Create directory if needed
    if not output_dir.exists():
        if create_directory:
            logger.info(f"Creating output directory: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(
                f"Output directory does not exist: {output_dir}. "
                f"Set create_directory=True to create it automatically."
            )

    # Dictionary to store saved file paths
    saved_files = {}
    total_files = len(clean_data_dict)

    logger.info(f"Starting to save {total_files} files as {file_format.upper()} to {output_dir}")

    # Process each ticker in the dictionary
    for idx, (filename, data_tuple) in enumerate(clean_data_dict.items(), start=1):
        try:
            # Validate data structure
            if not isinstance(data_tuple, tuple) or len(data_tuple) < 1:
                logger.warning(
                    f"Skipping {filename}: Expected tuple with at least 1 element, "
                    f"got {type(data_tuple)}"
                )
                continue

            # Extract ticker data (first element of tuple)
            ticker_data = data_tuple[0]

            # Validate ticker data type
            if not isinstance(ticker_data, (pl.LazyFrame, pl.DataFrame)):
                logger.warning(
                    f"Skipping {filename}: Expected LazyFrame or DataFrame, "
                    f"got {type(ticker_data)}"
                )
                continue

            # Determine output filename (replace extension if needed)
            original_stem = Path(filename).stem  # e.g., "AAPL" from "AAPL.csv"
            if file_format == "csv":
                output_filename = f"{original_stem}.csv"
            else:  # parquet
                output_filename = f"{original_stem}.parquet"

            output_path = output_dir / output_filename

            # Check if file exists and handle overwrite
            if output_path.exists() and not overwrite:
                logger.info(f"Skipping {filename}: File already exists at {output_path}")
                saved_files[filename] = output_path
                continue

            # Collect LazyFrame if needed (DataFrames are already collected)
            if isinstance(ticker_data, pl.LazyFrame):
                df = ticker_data.collect()
            else:
                df = ticker_data

            # Save to file based on format
            if file_format == "csv":
                df.write_csv(output_path)
            else:  # parquet
                df.write_parquet(
                    output_path,
                    compression="snappy",  # Good balance of speed and compression
                    use_pyarrow=False  # Use native Polars writer
                )

            # Store the saved path
            saved_files[filename] = output_path

            # Log progress every 100 files
            if idx % 100 == 0:
                logger.info(f"Progress: {idx}/{total_files} files saved ({idx/total_files*100:.1f}%)")

        except Exception as e:
            logger.error(f"Error saving {filename}: {e}", exc_info=True)
            continue

    # Final summary
    success_count = len(saved_files)
    logger.info(
        f"Completed: {success_count}/{total_files} files saved successfully "
        f"({success_count/total_files*100:.1f}%)"
    )

    return saved_files


def save_corrected_data_by_format(
    clean_data_dict: Dict[str, Tuple[Union[pl.LazyFrame, pl.DataFrame], pl.LazyFrame]],
    output_directory: Union[str, Path],
    save_csv: bool = True,
    save_parquet: bool = False,
    **kwargs
) -> Dict[str, Dict[str, Path]]:
    """
    Save corrected fundamental data in multiple formats simultaneously.

    Convenience function to save data in both CSV and Parquet formats if needed.

    Parameters
    ----------
    clean_data_dict : Dict[str, Tuple[Union[pl.LazyFrame, pl.DataFrame], pl.LazyFrame]]
        Dictionary of clean LazyFrames from data_corrector.py.

    output_directory : Union[str, Path]
        Base directory for output files.
        Subdirectories will be created for each format if both are enabled.

    save_csv : bool, default=True
        If True, saves data as CSV files in output_directory/csv/

    save_parquet : bool, default=False
        If True, saves data as Parquet files in output_directory/parquet/

    **kwargs
        Additional keyword arguments passed to save_corrected_data()
        (e.g., create_directory, overwrite)

    Returns
    -------
    Dict[str, Dict[str, Path]]
        Dictionary with format as keys ("csv", "parquet") and saved file dictionaries as values.
        Example: {"csv": {filename: path, ...}, "parquet": {filename: path, ...}}

    Examples
    --------
    >>> # Save as both CSV and Parquet
    >>> results = save_corrected_data_by_format(
    ...     clean_data_dict=clean_lfs,
    ...     output_directory="../Output/CorrectedData",
    ...     save_csv=True,
    ...     save_parquet=True
    ... )
    >>> print(f"CSV files: {len(results['csv'])}")
    >>> print(f"Parquet files: {len(results['parquet'])}")
    """
    output_dir = Path(output_directory)
    results = {}

    if not save_csv and not save_parquet:
        logger.warning("Both save_csv and save_parquet are False. No files will be saved.")
        return results

    if save_csv:
        csv_dir = output_dir / "csv"
        logger.info(f"Saving CSV files to {csv_dir}")
        results["csv"] = save_corrected_data(
            clean_data_dict=clean_data_dict,
            output_directory=csv_dir,
            file_format="csv",
            **kwargs
        )

    if save_parquet:
        parquet_dir = output_dir / "parquet"
        logger.info(f"Saving Parquet files to {parquet_dir}")
        results["parquet"] = save_corrected_data(
            clean_data_dict=clean_data_dict,
            output_directory=parquet_dir,
            file_format="parquet",
            **kwargs
        )

    return results
