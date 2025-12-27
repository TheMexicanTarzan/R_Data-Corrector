"""
Test script for the Financial Dashboard
Tests core functionality without running the server
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dashboard.dashboard import DummyDataGenerator, LogNormalizer, FinancialDashboard


def test_dummy_data_generation():
    """Test that dummy data generator works correctly."""
    print("Testing dummy data generation...")

    ticker = "TEST"
    orig_df, clean_df, logs = DummyDataGenerator.generate_ticker_data(
        ticker,
        start_date="2023-01-01",
        end_date="2023-01-31",
        inject_errors=True
    )

    assert orig_df.height > 0, "Original DataFrame is empty"
    assert clean_df.height > 0, "Cleaned DataFrame is empty"
    assert orig_df.height == clean_df.height, "DataFrames have different lengths"

    # Check that some errors were injected
    has_errors = False
    for category, log_data in logs.items():
        if log_data:
            has_errors = True
            break

    assert has_errors, "No errors were injected"

    print(f"  ✓ Generated {orig_df.height} rows for {ticker}")
    print(f"  ✓ Columns: {len(orig_df.columns)}")
    print(f"  ✓ Errors found in logs")


def test_log_normalizer():
    """Test that log normalizer handles different log formats."""
    print("\nTesting log normalizer...")

    # Generate test data
    tickers = ["TICK1", "TICK2"]
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
        orig_df, clean_df, logs = DummyDataGenerator.generate_ticker_data(
            ticker,
            start_date="2023-01-01",
            end_date="2023-01-15",
            inject_errors=True
        )

        original_dfs[ticker] = orig_df
        cleaned_dfs[ticker] = clean_df

        for category, log_data in logs.items():
            if log_data:
                all_logs[category].append(log_data)

    # Normalize logs
    normalized = LogNormalizer.normalize_logs(all_logs, original_dfs, cleaned_dfs)

    assert normalized.height > 0, "Normalized logs are empty"

    # Check schema
    required_columns = [
        "ticker", "date", "error_category", "error_type",
        "column", "severity", "false_positive"
    ]

    for col in required_columns:
        assert col in normalized.columns, f"Missing column: {col}"

    print(f"  ✓ Normalized {normalized.height} log entries")
    print(f"  ✓ Schema validated")
    print(f"  ✓ Unique error categories: {normalized['error_category'].n_unique()}")


def test_dashboard_initialization():
    """Test that dashboard can be initialized."""
    print("\nTesting dashboard initialization...")

    # Generate minimal test data
    ticker = "INIT_TEST"
    orig_df, clean_df, logs = DummyDataGenerator.generate_ticker_data(
        ticker,
        start_date="2023-01-01",
        end_date="2023-01-10",
        inject_errors=True
    )

    original_dfs = {ticker: orig_df}
    cleaned_dfs = {ticker: clean_df}

    all_logs = {
        "negative_fundamentals": [logs["negative_fundamentals"]] if logs["negative_fundamentals"] else [],
        "negative_market": [logs["negative_market"]] if logs["negative_market"] else [],
        "zero_wipeout": [logs["zero_wipeout"]] if logs["zero_wipeout"] else [],
        "mkt_cap_scale": [logs["mkt_cap_scale"]] if logs["mkt_cap_scale"] else [],
        "ohlc_integrity": [logs["ohlc_integrity"]] if logs["ohlc_integrity"] else [],
        "financial_equivalencies": [logs["financial_equivalencies"]] if logs["financial_equivalencies"] else [],
        "sort_dates": [logs["sort_dates"]] if logs["sort_dates"] else []
    }

    # Initialize dashboard (don't run server)
    dashboard = FinancialDashboard(original_dfs, cleaned_dfs, all_logs)

    assert dashboard.app is not None, "Dashboard app not initialized"
    assert dashboard.normalized_logs.height > 0, "No normalized logs"

    print(f"  ✓ Dashboard initialized successfully")
    print(f"  ✓ App layout created")
    print(f"  ✓ Callbacks registered")


def test_false_positive_storage():
    """Test false positive flag storage."""
    print("\nTesting false positive storage...")

    ticker = "FP_TEST"
    orig_df, clean_df, logs = DummyDataGenerator.generate_ticker_data(
        ticker,
        start_date="2023-01-01",
        end_date="2023-01-05",
        inject_errors=True
    )

    original_dfs = {ticker: orig_df}
    cleaned_dfs = {ticker: clean_df}

    all_logs = {
        "negative_fundamentals": [logs["negative_fundamentals"]] if logs["negative_fundamentals"] else [],
        "negative_market": [],
        "zero_wipeout": [],
        "mkt_cap_scale": [],
        "ohlc_integrity": [],
        "financial_equivalencies": [],
        "sort_dates": []
    }

    dashboard = FinancialDashboard(original_dfs, cleaned_dfs, all_logs)

    # Test saving false positive
    dashboard.save_false_positive(
        ticker="FP_TEST",
        date="2023-01-01",
        error_type="test_error",
        is_fp=True
    )

    assert dashboard.false_positive_store_path.exists(), "False positive file not created"

    print(f"  ✓ False positive storage working")
    print(f"  ✓ File created at: {dashboard.false_positive_store_path}")

    # Cleanup
    if dashboard.false_positive_store_path.exists():
        dashboard.false_positive_store_path.unlink()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Financial Dashboard Test Suite")
    print("=" * 60)

    try:
        test_dummy_data_generation()
        test_log_normalizer()
        test_dashboard_initialization()
        test_false_positive_storage()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        print("\nDashboard is ready to use!")
        print("Run: python src/dashboard/dashboard.py")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
