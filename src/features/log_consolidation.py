

def consolidate_audit_logs(raw_logs: list) -> dict | list:
    """
    Consolidate audit logs from parallel processing.

    When processing multiple tickers in parallel, each ticker returns audit logs
    (often empty [] or {}). This function flattens and merges them appropriately.

    Args:
        raw_logs: List of audit logs from parallel_process_tickers

    Returns:
        Consolidated audit logs - either a single list or dict depending on structure

    Examples:
        [[], [], []] → []
        [[], [{"error": "..."}], []] → [{"error": "..."}]
        [{}, {"col1": [...]}, {}] → {"col1": [...]}
        [{"hard": [], "soft": []}, {"hard": [...], "soft": []}] → {"hard": [...], "soft": [...]}
    """
    if not raw_logs:
        return []

    # Check the structure of non-empty logs to determine consolidation strategy
    sample_non_empty = None
    for log in raw_logs:
        if log:  # Find first non-empty log
            sample_non_empty = log
            break

    # If all logs are empty, return appropriate empty structure
    if sample_non_empty is None:
        # Check if we have lists or dicts
        has_dict = any(isinstance(log, dict) for log in raw_logs)
        return {} if has_dict else []

    # Strategy 1: List of lists → flatten to single list
    if isinstance(sample_non_empty, list):
        consolidated = []
        for log in raw_logs:
            if isinstance(log, list) and log:
                consolidated.extend(log)
        return consolidated

    # Strategy 2: List of dicts → merge dicts
    if isinstance(sample_non_empty, dict):
        # Check if it's a hard/soft filter structure (financial_unequivalencies)
        has_hard_soft = "hard_filter_errors" in sample_non_empty or "soft_filter_warnings" in sample_non_empty

        if has_hard_soft:
            # Merge hard/soft filter logs
            consolidated = {"hard_filter_errors": [], "soft_filter_warnings": []}
            for log in raw_logs:
                if isinstance(log, dict):
                    if "hard_filter_errors" in log and log["hard_filter_errors"]:
                        consolidated["hard_filter_errors"].extend(log["hard_filter_errors"])
                    if "soft_filter_warnings" in log and log["soft_filter_warnings"]:
                        consolidated["soft_filter_warnings"].extend(log["soft_filter_warnings"])
            return consolidated
        else:
            # Merge dict with nested lists (negative_fundamentals structure)
            consolidated = {}
            for log in raw_logs:
                if isinstance(log, dict):
                    for key, value in log.items():
                        if key not in consolidated:
                            consolidated[key] = []
                        if isinstance(value, list) and value:
                            consolidated[key].extend(value)
            return consolidated

    # Fallback: return as-is
    return raw_logs