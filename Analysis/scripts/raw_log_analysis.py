import re
from collections import defaultdict


def parse_error_log(log_text: str) -> dict[str, list[str]]:
    """
    Parse FMP error log and return a dictionary mapping error types to tickers.

    Args:
        log_text: Raw error log text

    Returns:
        Dictionary with error types as keys and lists of tickers as values
    """
    errors_by_type = defaultdict(list)

    # Pattern to match error lines: [ERROR] TICKER skipping output...
    pattern = r'\[ERROR\]\s+(\S+)\s+skipping output.*?:\s*(.+?)(?=\[ERROR\]|\Z)'

    matches = re.findall(pattern, log_text, re.DOTALL)

    for ticker, error_msg in matches:
        # Normalize error message (strip whitespace, collapse to single line)
        error_type = ' '.join(error_msg.strip().split())

        # Categorize into broader error types
        if "not correctly sorted by date" in error_type:
            category = "FundamentalData rows not sorted by date"
        elif "Negative" in error_type and "shares_outstanding" in error_type:
            category = "Negative shares outstanding"
        elif "No data returned by unadjusted market data endpoint" in error_type:
            category = "No unadjusted market data"
        elif "low > high" in error_type:
            category = "Market data low > high"
        elif "Negative" in error_type and "low" in error_type:
            category = "Negative price value"
        else:
            category = error_type  # Keep original if no match

        errors_by_type[category].append(ticker)

    return dict(errors_by_type)


def print_summary(errors: dict[str, list[str]]) -> None:
    """Print a formatted summary of errors."""
    print("=" * 60)
    print("FMP ERROR LOG SUMMARY")
    print("=" * 60)

    for error_type, tickers in sorted(errors.items(), key=lambda x: -len(x[1])):
        print(f"\n{error_type}")
        print(f"  Count: {len(tickers)}")
        print(f"  Tickers: {', '.join(sorted(tickers))}")


if __name__ == "__main__":
    # Example usage with sample log
    sample_log = """[ERROR] RBA skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NPB skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] MTZ skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] FLS skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ALG skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SCI skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] DAN skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SPB skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RGA skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] HELE skipping output as it presented the following error during data assembly:
  Fundamental data processing error for date 2017-05-01: Negative FundamentalDataRowIncome.weighted_average_diluted_shares_outstanding detected
[ERROR] DG skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] WWW skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RZC skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AMKR skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] KOPN skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] DOV skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] IESC skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] JBL skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] KODK skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AGM skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] FBP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NPK skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AES skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] UDR skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] XELB skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] DLPN skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] MCHX skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] EVTV skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] QLGN skipping output as it presented the following error during data assembly:
  Fundamental data processing error for date 2025-11-14: Negative FundamentalDataRowIncome.weighted_average_basic_shares_outstanding detected
[ERROR] EYPT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SOTK skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] TPCS skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] VRNT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NEPH skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SQFT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] LIVE skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SPOK skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NBP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] HIVE skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYN skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ELDN skipping output as it presented the following error during data assembly:
  Fundamental data processing error for date 2025-11-14: Negative FundamentalDataRowIncome.weighted_average_diluted_shares_outstanding detected
[ERROR] RILYZ skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ATON skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYL skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] TRS skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] VSAT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYG skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] POWI skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SQFTP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILYK skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RCAT skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] RILY skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NTRP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AGM-PF skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AGM-PD skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AGM-PG skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ASB-PF skipping output as it presented the following error during data assembly:
  Market data processing error: date: 1993-06-01: Negative MarketDataDailyRow.low for date 1993-06-01
[ERROR] AGM-PE skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] UWMC-WT skipping output as it presented the following error during data assembly:
  Market data processing error: date: 2023-07-07: MarketDataDailyRow low > high for date 2023-07-07
[ERROR] TDACW skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] BFLY-WT skipping output as it presented the following error during data assembly:
  Market data processing error: date: 2022-08-11: MarketDataDailyRow low > high for date 2022-08-11
[ERROR] SQFTW skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] AGM-A skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] XOMAP skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] PETVW skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ZIVOW skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] FARO skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] APDN skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] STAF skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] PEI-PB skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] LRFC skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] PEI-PD skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] PSB-PX skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] INVO skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] SASR skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] PNC-PP skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] IDEX skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] MICS skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ML-WT skipping output as it presented the following error during data assembly:
  Market data processing error: date: 2023-03-03: MarketDataDailyRow low > high for date 2023-03-03
[ERROR] IVAC skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] NRZ-PC skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] MSON skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] ALP-PQ skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] NRZ-PB skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] AGM-PC skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] STZ-B skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] FOE skipping output as it presented the following error during data assembly:
  Fundamental data processing error: FundamentalData.rows are not correctly sorted by date, this usually indicates missing or amended statements
[ERROR] PSB-PZ skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] PEI-PC skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] PSB-PY skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
[ERROR] NRZ-PA skipping output as it presented the following error during data assembly:
  Market data processing error: No data returned by unadjusted market data endpoint
    """

    result = parse_error_log(sample_log)
    print_summary(result)

    # print("\n\nRaw dictionary output:")
    # print(result)