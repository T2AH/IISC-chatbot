"""
Logging configuration for the IISc chatbot project
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    Setup logging configuration using loguru
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    # Remove default handler
    logger.remove()
    
    # Add console handler with color
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level=log_level,
            rotation="500 MB",
            retention="30 days",
            compression="zip"
        )
    
    logger.info(f"Logging initialized at {log_level} level")


def get_logger(name: str):
    """
    Get a logger instance for a specific module
    
    Args:
        name: Module name
    
    Returns:
        Logger instance
    """
    return logger.bind(name=name)
