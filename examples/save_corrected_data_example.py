"""
Example script demonstrating how to use the save_corrected_data function.

This shows different ways to save corrected fundamental data to CSV or Parquet files.
"""
from pathlib import Path
import sys

# Add parent directory to path to import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.output_handlers import save_corrected_data, save_corrected_data_by_format


def example_basic_csv_save(clean_data_dict, output_dir):
    """
    Example 1: Basic CSV save
    """
    print("\n=== Example 1: Basic CSV Save ===")
    saved_files = save_corrected_data(
        clean_data_dict=clean_data_dict,
        output_directory=output_dir / "csv_output",
        file_format="csv",
        create_directory=True,
        overwrite=True
    )
    print(f"Saved {len(saved_files)} CSV files")
    return saved_files


def example_parquet_save(clean_data_dict, output_dir):
    """
    Example 2: Parquet save (smaller files, faster I/O)
    """
    print("\n=== Example 2: Parquet Save ===")
    saved_files = save_corrected_data(
        clean_data_dict=clean_data_dict,
        output_directory=output_dir / "parquet_output",
        file_format="parquet",
        create_directory=True,
        overwrite=True
    )
    print(f"Saved {len(saved_files)} Parquet files")
    return saved_files


def example_dual_format_save(clean_data_dict, output_dir):
    """
    Example 3: Save in both CSV and Parquet formats
    """
    print("\n=== Example 3: Dual Format Save ===")
    results = save_corrected_data_by_format(
        clean_data_dict=clean_data_dict,
        output_directory=output_dir / "dual_format_output",
        save_csv=True,
        save_parquet=True,
        create_directory=True,
        overwrite=True
    )
    print(f"Saved {len(results['csv'])} CSV files")
    print(f"Saved {len(results['parquet'])} Parquet files")
    return results


def example_no_overwrite(clean_data_dict, output_dir):
    """
    Example 4: Skip existing files (no overwrite)
    """
    print("\n=== Example 4: No Overwrite (Skip Existing) ===")
    saved_files = save_corrected_data(
        clean_data_dict=clean_data_dict,
        output_directory=output_dir / "no_overwrite_output",
        file_format="csv",
        create_directory=True,
        overwrite=False  # Will skip files that already exist
    )
    print(f"Processed {len(saved_files)} files (skipped existing)")
    return saved_files


def main():
    """
    Main function to run all examples.

    Note: This requires having clean_data_dict from data_corrector.py
    In practice, you would run this after data cleaning is complete.
    """
    print("=" * 60)
    print("Save Corrected Data Examples")
    print("=" * 60)

    # In a real scenario, you would get clean_lfs from data_corrector.py
    # For this example, we'll just show how to call the functions

    print("\nThese examples show how to use save_corrected_data function.")
    print("In your data_corrector.py workflow, call it like this:")
    print()
    print("# After running sanity checks and statistical filters:")
    print("from src.output_handlers import save_corrected_data")
    print()
    print("# Save as CSV")
    print("saved_files = save_corrected_data(")
    print("    clean_data_dict=clean_lfs,")
    print("    output_directory='../Output/CorrectedData',")
    print("    file_format='csv'")
    print(")")
    print()
    print("# Or save as Parquet for better compression")
    print("saved_files = save_corrected_data(")
    print("    clean_data_dict=clean_lfs,")
    print("    output_directory='../Output/CorrectedData',")
    print("    file_format='parquet'")
    print(")")
    print()
    print("# The function is already integrated in data_corrector.py")
    print("# Set save_data = True to enable automatic saving")


if __name__ == "__main__":
    main()
