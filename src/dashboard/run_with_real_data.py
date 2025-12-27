"""
Integration example: Run the dashboard with real data from data_corrector.py

This script shows how to integrate the Financial Dashboard with the actual
cleaning pipeline output.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import polars as pl
from src.dashboard.dashboard import FinancialDashboard


def load_real_data():
    """
    Load real data from the cleaning pipeline.

    This is a template - adjust the imports and paths based on your setup.
    """
    print("Loading data from cleaning pipeline...")

    try:
        # Option 1: Import from data_corrector module
        from src.data_corrector import (
            dataframe_dict,
            dataframe_dict_clean_financial_equivalencies,
            negative_fundamentals_logs,
            negative_market_logs,
            zero_wipeout_logs,
            shares_outstanding_logs,
            ohlc_logs,
            financial_unequivalencies_logs,
            unsorted_dates_logs
        )

        # Collect LazyFrames into DataFrames
        print("  Converting LazyFrames to DataFrames...")
        original_dfs = {}
        cleaned_dfs = {}

        for ticker in dataframe_dict.keys():
            # Collect original data
            original_dfs[ticker] = dataframe_dict[ticker].collect()

            # Collect cleaned data
            if ticker in dataframe_dict_clean_financial_equivalencies:
                cleaned_dfs[ticker] = dataframe_dict_clean_financial_equivalencies[ticker].collect()

        # Organize logs
        all_logs = {
            "negative_fundamentals": negative_fundamentals_logs,
            "negative_market": negative_market_logs,
            "zero_wipeout": zero_wipeout_logs,
            "mkt_cap_scale": shares_outstanding_logs,
            "ohlc_integrity": ohlc_logs,
            "financial_equivalencies": financial_unequivalencies_logs,
            "sort_dates": unsorted_dates_logs
        }

        print(f"  ✓ Loaded {len(original_dfs)} tickers")
        print(f"  ✓ Total rows: {sum(df.height for df in original_dfs.values())}")

        return original_dfs, cleaned_dfs, all_logs

    except ImportError as e:
        print(f"  ✗ Could not import from data_corrector: {e}")
        print("\n  Alternative: Run data_corrector.py first to generate data,")
        print("  or use the dummy data generator (see main() in dashboard.py)")
        return None, None, None


def load_from_saved_files(data_dir: Path):
    """
    Load data from saved Parquet/CSV files.

    Args:
        data_dir: Directory containing saved data files

    Returns:
        Tuple of (original_dfs, cleaned_dfs, all_logs)
    """
    print(f"Loading data from files in {data_dir}...")

    original_dfs = {}
    cleaned_dfs = {}

    # Load original data
    original_dir = data_dir / "original"
    if original_dir.exists():
        for file in original_dir.glob("*.parquet"):
            ticker = file.stem
            original_dfs[ticker] = pl.read_parquet(file)

    # Load cleaned data
    cleaned_dir = data_dir / "cleaned"
    if cleaned_dir.exists():
        for file in cleaned_dir.glob("*.parquet"):
            ticker = file.stem
            cleaned_dfs[ticker] = pl.read_parquet(file)

    # Load logs (if saved as JSON)
    import json

    all_logs = {
        "negative_fundamentals": [],
        "negative_market": [],
        "zero_wipeout": [],
        "mkt_cap_scale": [],
        "ohlc_integrity": [],
        "financial_equivalencies": [],
        "sort_dates": []
    }

    logs_file = data_dir / "logs.json"
    if logs_file.exists():
        with open(logs_file, 'r') as f:
            all_logs = json.load(f)

    print(f"  ✓ Loaded {len(original_dfs)} original dataframes")
    print(f"  ✓ Loaded {len(cleaned_dfs)} cleaned dataframes")

    return original_dfs, cleaned_dfs, all_logs


def main():
    """Main function to run dashboard with real data."""

    print("=" * 60)
    print("Financial Dashboard - Real Data Integration")
    print("=" * 60)
    print()

    # Try to load real data
    original_dfs, cleaned_dfs, all_logs = load_real_data()

    # If real data not available, try loading from files
    if original_dfs is None:
        data_dir = Path("data")  # Adjust this path
        if data_dir.exists():
            original_dfs, cleaned_dfs, all_logs = load_from_saved_files(data_dir)

    # If still no data, fall back to dummy data
    if original_dfs is None or len(original_dfs) == 0:
        print("\nNo real data available. Generating dummy data...")
        print("(To use real data, run data_corrector.py first)")
        print()

        from src.dashboard.dashboard import DummyDataGenerator

        tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        original_dfs = {}
        cleaned_dfs = {}
        all_logs = {
            "negative_fundamentals": [],
            "negative_market": [],
            "zero_wipeout": [],
            "mkt_cap_scale": [],
            "ohlc_integrity": [],
            "financial_equivalencies": [],
            "sort_dates": []
        }

        for ticker in tickers:
            print(f"  Generating data for {ticker}...")
            orig_df, clean_df, logs = DummyDataGenerator.generate_ticker_data(
                ticker,
                start_date="2020-01-01",
                end_date="2023-12-31",
                inject_errors=True
            )

            original_dfs[ticker] = orig_df
            cleaned_dfs[ticker] = clean_df

            for category, log_data in logs.items():
                if log_data:
                    all_logs[category].append(log_data)

        print(f"\n  ✓ Generated dummy data for {len(tickers)} tickers")

    # Create and run dashboard
    print("\n" + "=" * 60)
    print("Initializing Dashboard...")
    print("=" * 60)
    print()

    dashboard = FinancialDashboard(original_dfs, cleaned_dfs, all_logs)

    print("✓ Dashboard initialized successfully")
    print()
    print("=" * 60)
    print("Dashboard is running at: http://localhost:8050")
    print("=" * 60)
    print()
    print("Features:")
    print("  • Filter by ticker and error category")
    print("  • View detailed error logs in AG Grid table")
    print("  • Compare original vs cleaned data in time series")
    print("  • Flag false positives")
    print("  • Special charts for accounting equation mismatches")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # Run the dashboard
    dashboard.run(debug=True, port=8050)


if __name__ == "__main__":
    main()
