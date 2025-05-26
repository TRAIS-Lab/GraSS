"""
GradComp - Gradient Component-Based Influence Attribution
"""

import logging
import sys

def setup_logger(name=__name__, level=logging.INFO, log_file=None):
    """Configure and return a logger with the specified settings."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    # Create console handler with a proper formatter
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt="%m/%d/%Y %H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Add file handler if specified
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

# Default logger for the package
logger = setup_logger()
logger.propagate = False

# Package version
__version__ = '0.1.0'

__all__ = [
    'setup_logger',
    'logger'
]