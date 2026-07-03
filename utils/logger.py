"""
Centralized logging for HexaShop SCM agents.
Separate log files per agent/module.
"""
import logging
import sys
from pathlib import Path
from utils.config import LOG_LEVEL, LOG_DIR, FILE_LOGGING

# Default log formats
_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Create logs directory if file logging is enabled
if FILE_LOGGING:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cache for handlers to avoid duplicate file/console handlers
_file_handlers = {}
_error_handler = None
_console_handler = None


def _get_console_handler() -> logging.StreamHandler:
    """Retrieve or create the cached console stream handler."""
    global _console_handler
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
        _console_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    return _console_handler


def _get_file_handler(file_path: Path) -> logging.FileHandler:
    """Retrieve or create a cached file handler for a specific file path."""
    path_str = str(file_path.resolve())
    if path_str not in _file_handlers:
        handler = logging.FileHandler(path_str, encoding="utf-8")
        handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
        _file_handlers[path_str] = handler
    return _file_handlers[path_str]


def _get_error_handler(file_path: Path) -> logging.FileHandler:
    """Retrieve or create the cached aggregated error file handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = logging.FileHandler(str(file_path.resolve()), encoding="utf-8")
        _error_handler.setLevel(logging.ERROR)
        _error_handler.setFormatter(logging.Formatter(_FORMAT, _DATE_FORMAT))
    return _error_handler


def _map_logger_name_to_filename(name: str) -> str:
    """Map a logger name to its corresponding log file prefix."""
    if not name:
        return "app"

    # Match specific agents
    for agent in [
        "inventory_agent",
        "forecasting_agent",
        "procurement_agent",
        "logistics_agent",
        "customer_comms_agent",
        "supervisor_agent"
    ]:
        if agent in name:
            return agent

    # Match tools
    if name.startswith("tools") or "tools" in name:
        return "tools"

    # Match graph / orchestration
    if "graph" in name or "orchestration" in name or "langgraph" in name:
        return "graph"

    return "app"


def get_logger(name: str) -> logging.Logger:
    """Get a logger with both console and file handlers.

    Args:
        name: Logger name (e.g., 'inventory_agent', 'procurement_agent')

    Returns:
        logging.Logger: Configured logger instance

    Console logs go to stdout. File logs go to logs/{mapped_name}.log.
    All ERROR level logs also go to logs/errors.log.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    # Add console handler if not already present
    has_console = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        logger.addHandler(_get_console_handler())

    # Add file handlers if enabled
    if FILE_LOGGING:
        file_prefix = _map_logger_name_to_filename(name)
        
        # Mapped agent/module log handler
        agent_file_path = LOG_DIR / f"{file_prefix}.log"
        file_handler = _get_file_handler(agent_file_path)
        if file_handler not in logger.handlers:
            logger.addHandler(file_handler)

        # Aggregated error log handler
        error_file_path = LOG_DIR / "errors.log"
        error_handler = _get_error_handler(error_file_path)
        if error_handler not in logger.handlers:
            logger.addHandler(error_handler)

    return logger


def get_all_log_paths() -> dict:
    """Get paths of all expected log files.

    Returns:
        dict: Mapping of log keys to their Path locations.
    """
    return {
        "app": LOG_DIR / "app.log",
        "inventory_agent": LOG_DIR / "inventory_agent.log",
        "forecasting_agent": LOG_DIR / "forecasting_agent.log",
        "procurement_agent": LOG_DIR / "procurement_agent.log",
        "logistics_agent": LOG_DIR / "logistics_agent.log",
        "customer_comms_agent": LOG_DIR / "customer_comms_agent.log",
        "supervisor_agent": LOG_DIR / "supervisor_agent.log",
        "graph": LOG_DIR / "graph.log",
        "tools": LOG_DIR / "tools.log",
        "errors": LOG_DIR / "errors.log",
    }
