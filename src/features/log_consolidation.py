

def consolidate_audit_logs(raw_logs: list) -> dict:
    """
    Consolidate audit logs from parallel processing and add summary statistics.

    When processing multiple tickers in parallel, each ticker returns audit logs
    (often empty [] or {}). This function flattens and merges them appropriately,
    and wraps the result with a summary containing total counts.

    Args:
        raw_logs: List of audit logs from parallel_process_tickers

    Returns:
        Dictionary with "summary" and "data" keys:
        {
            "summary": {
                "total_logs": int,
                "breakdown": {...}  # Structure depends on log type
            },
            "data": {...}  # The actual consolidated logs
        }

    Examples:
        [[], [], []] → {"summary": {"total_logs": 0}, "data": []}
        [[], [{"error": "..."}], []] → {"summary": {"total_logs": 1}, "data": [{"error": "..."}]}
        [{}, {"col1": [...]}, {}] → {"summary": {"total_logs": N, "breakdown": {"col1": N}}, "data": {"col1": [...]}}
    """
    if not raw_logs:
        return {"summary": {"total_logs": 0}, "data": []}

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
        return {"summary": {"total_logs": 0}, "data": {} if has_dict else []}

    # Strategy 1: List of lists → flatten to single list
    if isinstance(sample_non_empty, list):
        consolidated = []
        for log in raw_logs:
            if isinstance(log, list) and log:
                consolidated.extend(log)

        total_logs = len(consolidated)
        return {
            "summary": {
                "total_logs": total_logs
            },
            "data": consolidated
        }

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

            # Calculate summary
            hard_count = len(consolidated["hard_filter_errors"])
            soft_count = len(consolidated["soft_filter_warnings"])
            total_logs = hard_count + soft_count

            return {
                "summary": {
                    "total_logs": total_logs,
                    "breakdown": {
                        "hard_filter_errors": hard_count,
                        "soft_filter_warnings": soft_count
                    }
                },
                "data": consolidated
            }
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

            # Calculate summary
            breakdown = {key: len(value) for key, value in consolidated.items()}
            total_logs = sum(breakdown.values())

            return {
                "summary": {
                    "total_logs": total_logs,
                    "breakdown": breakdown
                },
                "data": consolidated
            }

    # Fallback: return as-is with summary
    total_logs = len(raw_logs) if isinstance(raw_logs, list) else 1
    return {
        "summary": {
            "total_logs": total_logs
        },
        "data": raw_logs
    }