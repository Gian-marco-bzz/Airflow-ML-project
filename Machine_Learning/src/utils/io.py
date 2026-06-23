"""
Utilities for file I/O and configuration management.

Includes:
- load_config(): Read YAML configuration
- ensure_dir(): Create directories recursively
- get_logger(): Setup logging with consistent format
"""

import os
import logging
import yaml


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file with error handling.

    Args:
        config_path: Path to config.yaml

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is malformed
    """
    logger = logging.getLogger(__name__)

    logger.debug(f"Loading config from {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            logger.info(f"✓ Config loaded ({len(cfg)} top-level keys)")
            return cfg

    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise

    except yaml.YAMLError as e:
        logger.error(f"YAML parsing error in {config_path}: {e}")
        raise


def ensure_dir(path: str) -> None:
    """
    Create directory recursively if it doesn't exist.

    Args:
        path: Directory path to create
    """
    os.makedirs(path, exist_ok=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """
    Get logger with consistent format.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process...")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set line up
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(levelname)s %(name)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return logger