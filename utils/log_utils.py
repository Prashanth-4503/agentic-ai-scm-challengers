"""
Helper utilities for SCM agent log management.
Provides tools to view log paths, clear logs, and summarize errors.
"""
import re
from pathlib import Path
from utils.logger import get_all_log_paths


def get_log_file_paths() -> dict:
    """Returns a dictionary of all log file keys and their absolute file paths as strings.

    Returns:
        dict: Mapping of log keys to absolute path strings.
    """
    return {k: str(v.resolve()) for k, v in get_all_log_paths().items()}


def clear_all_logs() -> None:
    """Clears/empties all configured log files by truncating them to 0 bytes."""
    for path in get_all_log_paths().values():
        if path.exists():
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.truncate(0)
            except Exception:
                pass  # Ignore files that are currently locked or open


def get_error_summary() -> dict:
    """Reads the aggregated errors log and returns a structured summary.

    Supports multiline error tracebacks by grouping trailing lines under the last seen error.

    Returns:
        dict: A summary dictionary:
            - total_errors (int): Total count of logged errors.
            - errors_by_module (dict): Counts keyed by originating logger name.
            - recent_errors (list): List of dicts containing the last 10 errors.
    """
    error_log_path = get_all_log_paths()["errors"]
    summary = {
        "total_errors": 0,
        "errors_by_module": {},
        "recent_errors": []
    }

    if not error_log_path.exists():
        return summary

    try:
        with open(error_log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current_error = None
        # Format matches: [HH:MM:SS] ERROR    logger_name — Log message
        log_pattern = re.compile(r"^\[(.*?)\]\s+(ERROR)\s+(\S+)\s+—\s+(.*)$")

        for line in lines:
            line_stripped = line.rstrip()
            if not line_stripped:
                continue

            match = log_pattern.match(line_stripped)
            if match:
                if current_error:
                    summary["recent_errors"].append(current_error)

                timestamp, level, module, message = match.groups()
                current_error = {
                    "timestamp": timestamp,
                    "level": level,
                    "module": module,
                    "message": message
                }
                summary["total_errors"] += 1
                summary["errors_by_module"][module] = summary["errors_by_module"].get(module, 0) + 1
            else:
                # Continuation of multiline tracebacks
                if current_error:
                    current_error["message"] += "\n" + line_stripped

        if current_error:
            summary["recent_errors"].append(current_error)

        # Truncate to last 10 errors for brevity, reversed to show most recent first
        summary["recent_errors"] = list(reversed(summary["recent_errors"][-10:]))

    except Exception as e:
        summary["parsing_error"] = str(e)

    return summary
