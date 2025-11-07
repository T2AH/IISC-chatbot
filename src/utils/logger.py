import logging
import os
from datetime import datetime
from pathlib import Path

def setup_logger(name, log_file=None, level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    Path('logs').mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"logs/{name}_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
