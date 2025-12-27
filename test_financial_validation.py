"""
Test script for validate_financial_equivalencies function
Demonstrates hard and soft filter validations with example data
"""

import polars
from src.modules.errors.sanity_check.sanity_check import validate_financial_equivalencies


def create_test_data():
    """Create sample financial data with intentional errors"""
    return polars.DataFrame({
        "m_date": ["2023-01-01", "2023-02-01", "2023-03-01", "2023-04-01", "2023-05-01"],

        # Hard Filter Test 1: Assets (violation in row 0 and 1)
        "fbs_assets": [1000.0, 2000.0, 3000.0, 0.0, 500.0],
        "fbs_current_assets": [400.0, 800.0, 1200.0, 0.0, 0.0],  # Row 1: sum = 1600 != 2000
        "fbs_noncurrent_assets": [500.0, 800.0, 1800.0, 0.0, 0.0],  # Row 4: edge case

        # Hard Filter Test 2: Liabilities (violation in row 2)
        "fbs_liabilities": [500.0, 1000.0, 1500.0, 600.0, 250.0],
        "fbs_current_liabilities": [200.0, 400.0, 500.0, 300.0, 125.0],  # Row 2: sum = 1200 != 1500
        "fbs_noncurrent_liabilities": [300.0, 600.0, 700.0, 300.0, 125.0],

        # Soft Filter Test 1: Equity (violation in row 1 and 3)
        "fbs_stockholder_equity": [500.0, 1000.0, 1500.0, 1400.0, 250.0],
        "fbs_common_stock_value": [100.0, 200.0, 300.0, 300.0, 50.0],
        "fbs_additional_paid_in_capital": [200.0, 400.0, 600.0, 600.0, 100.0],
        "fbs_retained_earnings": [150.0, 300.0, 450.0, 450.0, 75.0],  # Row 1: sum = 950 != 1000
        "fbs_other_stockholder_equity": [50.0, 100.0, 150.0, 150.0, 25.0],  # Row 3: sum = 1500 != 1400

        # Soft Filter Test 2: Cash (violation in row 2 and 4)
        "fcf_period_end_cash": [100.0, 200.0, 305.0, 400.0, 55.0],  # Row 2: 305 != 300
        "fbs_cash_and_cash_equivalents": [100.0, 200.0, 300.0, 400.0, 50.0],  # Row 4: 55 != 50
    })


def main():
    print("=" * 80)
    print("Testing validate_financial_equivalencies Function")
    print("=" * 80)

    # Create test data
    df = create_test_data()

    print("\n--- ORIGINAL DATA ---")
    print(df)

    # Run validation with tolerance = 1.0
    cleaned_df, error_log = validate_financial_equivalencies(
        df=df,
        ticker="TEST",
        date_col="m_date",
        tolerance=1.0
    )

    print("\n--- CLEANED DATA ---")
    print(cleaned_df)

    print("\n--- ERROR LOG ---")
    print(f"\nHard Filter Errors: {len(error_log['hard_filter_errors'])} found")
    for i, error in enumerate(error_log['hard_filter_errors'], 1):
        print(f"\n  Error {i}:")
        print(f"    Date: {error['date']}")
        print(f"    Type: {error['error_type']}")
        print(f"    Total: {error['total']:.2f}")
        print(f"    Current: {error['current']:.2f} -> {error['corrected_current']:.2f}")
        print(f"    Noncurrent: {error['noncurrent']:.2f} -> {error['corrected_noncurrent']:.2f}")
        print(f"    Method: {error['correction_method']}")
        print(f"    Difference: {error['difference']:.2f}")

    print(f"\nSoft Filter Warnings: {len(error_log['soft_filter_warnings'])} found")
    for i, warning in enumerate(error_log['soft_filter_warnings'], 1):
        print(f"\n  Warning {i}:")
        print(f"    Date: {warning['date']}")
        print(f"    Type: {warning['error_type']}")
        if warning['error_type'] == 'equity_mismatch':
            print(f"    Total: {warning['total']:.2f}")
            print(f"    Component Sum: {warning['component_sum']:.2f}")
            print(f"    Difference: {warning['difference']:.2f}")
        elif warning['error_type'] == 'cash_mismatch':
            print(f"    FCF Period End Cash: {warning['fcf_period_end_cash']:.2f}")
            print(f"    Cash and Equivalents: {warning['fbs_cash_and_cash_equivalents']:.2f}")
            print(f"    Difference: {warning['difference']:.2f}")

    print("\n--- DATA WARNING FLAGS ---")
    print(cleaned_df.select(["m_date", "data_warning"]))

    print("\n--- VERIFICATION ---")
    # Verify hard filter corrections
    verification = cleaned_df.select([
        "m_date",
        "fbs_assets",
        (polars.col("fbs_current_assets") + polars.col("fbs_noncurrent_assets")).alias("assets_sum"),
        (polars.col("fbs_assets") - (polars.col("fbs_current_assets") + polars.col("fbs_noncurrent_assets"))).abs().alias("assets_diff"),
        "fbs_liabilities",
        (polars.col("fbs_current_liabilities") + polars.col("fbs_noncurrent_liabilities")).alias("liabilities_sum"),
        (polars.col("fbs_liabilities") - (polars.col("fbs_current_liabilities") + polars.col("fbs_noncurrent_liabilities"))).abs().alias("liabilities_diff"),
    ])
    print("\nAll hard filter differences should be <= tolerance (1.0):")
    print(verification)

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
