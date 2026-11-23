"""Shared helpers for the trading agent."""

import os
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv("DEBUG", "true").lower() in {"1", "true", "yes", "on"}

def debug(message: str) -> None:
    """Print a debug message when debug logging is enabled."""
    if DEBUG:
        print(f"[DEBUG] {message}")


