import logging
import sys
from pathlib import Path
from typing import Optional
import tqdm

class TqdmLoggingHandler(logging.Handler):
    """
    A logging handler that outputs log messages through tqdm.write(),
    avoiding interference with tqdm progress bars.
    """
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(
    run_dir: Optional[Path] = None,
    log_file: str = "pipeline.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup centralized logging for Demandify.
    
    Args:
        run_dir: Directory to save log file (if None, only console logging)
        log_file: Name of the log file
        level: Logging level
        
    Returns:
        The configured root logger
    """
    # Get the library root logger
    logger = logging.getLogger("demandify")
    logger.setLevel(level)
    
    # Remove existing handlers to prevent duplication ("Triple logging" fix)
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # Standard formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 1. Console Handler (Tqdm-aware)
    # We use TqdmLoggingHandler to print logs cleanly even when progress bars are active
    console_handler = TqdmLoggingHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 2. File Handler (if run_dir provided)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / log_file
        
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Log initialization message (only goes to file initially if level > INFO)
        logger.debug(f"Logging initialized. File: {log_path}")
        
    # Prevent propagation to root logger (avoid double logging with system rules)
    logger.propagate = False
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a child logger of 'demandify'."""
    # Ensure name starts with demandify prefix if it's a sub-module
    if not name.startswith("demandify"):
        name = f"demandify.{name}"
    return logging.getLogger(name)
