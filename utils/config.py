"""
Central configuration — reads all settings from .env once at import time.
"""
import os
from dotenv import load_dotenv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)

# Azure OpenAI — support both KEY and API_KEY env var names
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# Business thresholds
PO_APPROVAL_THRESHOLD = int(os.getenv("PO_APPROVAL_THRESHOLD", "50000"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
FILE_LOGGING = os.getenv("FILE_LOGGING", "true").strip().lower() in ("1", "true", "yes")

# Base log directory configuration
LOG_DIR_PATH = os.getenv("LOG_DIR", "logs")
if Path(LOG_DIR_PATH).is_absolute():
    LOG_DIR = Path(LOG_DIR_PATH)
else:
    LOG_DIR = BASE_DIR / LOG_DIR_PATH

# Show full CrewAI agent/task reasoning traces in the console (nice for demos,
# noisy for normal runs) — toggle with VERBOSE_AGENTS=true in .env
VERBOSE_AGENTS = os.getenv("VERBOSE_AGENTS", "false").strip().lower() in ("1", "true", "yes")


def validate_config():
    """Raise if critical Azure settings are missing."""
    missing = []
    if not AZURE_OPENAI_ENDPOINT:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not AZURE_OPENAI_API_KEY:
        missing.append("AZURE_OPENAI_API_KEY / AZURE_OPENAI_KEY")
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")


def get_llm():
    """Return a configured AzureChatOpenAI instance (reusable across agents)."""
    from langchain_openai import AzureChatOpenAI
    validate_config()
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
        api_version=AZURE_OPENAI_API_VERSION,
        temperature=0,
    )