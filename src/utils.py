# This file contains helper utilities for directory verification, path resolution, and logging setup, used across the project to ensure cross-platform compatibility.

"""
====================================================
Module : utils.py
Project: Airline Passenger Forecasting
Purpose: Utility functions for path resolution and setup
====================================================
"""

import logging
from pathlib import Path

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AirlinePassengerForecasting")


def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory.
    Returns:
        Path: Project root directory
    """
    # This file is in src/utils.py, so parent of parent is the project root.
    return Path(__file__).resolve().parent.parent


def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root.
    Args:
        relative_path (str): Path relative to the project root
    Returns:
        Path: Resolved absolute path
    """
    return get_project_root() / relative_path


def ensure_directories():
    """
    Create necessary project directories if they do not exist.
    """
    root = get_project_root()
    for folder in ["data", "models", "outputs"]:
        path = root / folder
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")
