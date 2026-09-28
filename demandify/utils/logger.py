import contextvars
import logging
from pathlib import Path
from typing import Optional
import tqdm

current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "current_run_id", default=None
)


class RunFilter(logging.Filter):
    """Filter log records so run-specific file handlers only receive records from their run."""

    def __init__(self, run_id: Optional[str] = None):
        super().__init__()
        self.run_id = run_id

    def filter(self, record: logging.LogRecord) -> bool:
        if self.run_id is None:
            return True
        active_id = getattr(record, "run_id", None) or current_run_id.get()
        if active_id is None:
            return True
        return active_id == self.run_id


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
    level: int = logging.INFO,
    run_id: Optional[str] = None,
) -> logging.Logger:
    """
    Setup centralized logging for Demandify.

    Args:
        run_dir: Directory to save log file (if None, only console logging)
        log_file: Name of the log file
        level: Logging level
        run_id: Optional run ID for run-scoped file logging

    Returns:
        The configured root logger
    """
    # Get the library root logger
    logger = logging.getLogger("demandify")
    logger.setLevel(level)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 1. Console Handler (Tqdm-aware) - ensure only one is added
    has_console = any(isinstance(h, TqdmLoggingHandler) for h in logger.handlers)
    if not has_console:
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 2. File Handler (if run_dir provided)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = (run_dir / log_file).resolve()

        # Remove existing file handler for the exact same path if present
        for h in list(logger.handlers):
            if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path:
                logger.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(logging.DEBUG)  # Always capture everything in file
        file_handler.setFormatter(formatter)
        if run_id:
            file_handler.addFilter(RunFilter(run_id))
        logger.addHandler(file_handler)

        logger.debug(f"Logging initialized. File: {log_path}")

    # Prevent propagation to root logger (avoid double logging with system rules)
    logger.propagate = False

    return logger


def remove_run_logging(run_dir: Path, log_file: str = "pipeline.log") -> None:
    """Remove and close file handler for a specific run directory."""
    logger = logging.getLogger("demandify")
    log_path = (run_dir / log_file).resolve()
    for h in list(logger.handlers):
        if isinstance(h, logging.FileHandler) and Path(h.baseFilename).resolve() == log_path:
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

