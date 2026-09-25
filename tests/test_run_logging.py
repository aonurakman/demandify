"""Tests for concurrent and isolated run logging behavior."""

import logging
from demandify.utils.logger import (
    current_run_id,
    remove_run_logging,
    setup_logging,
    TqdmLoggingHandler,
)


def test_concurrent_run_logging_isolation(tmp_path):
    """Multiple runs do not clobber each other's handlers and log messages are isolated."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    # Setup run A
    logger = setup_logging(run_dir=dir_a, run_id="run_a", log_file="pipeline.log")
    file_handlers_a = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers_a) >= 1

    # Setup run B — should NOT wipe run A's file handler
    setup_logging(run_dir=dir_b, run_id="run_b", log_file="pipeline.log")
    file_handlers_both = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers_both) >= 2

    # Log in run_a context
    token_a = current_run_id.set("run_a")
    logger.info("Message for run A")
    current_run_id.reset(token_a)

    # Log in run_b context
    token_b = current_run_id.set("run_b")
    logger.info("Message for run B")
    current_run_id.reset(token_b)

    # Read both log files
    log_a = (dir_a / "pipeline.log").read_text(encoding="utf-8")
    log_b = (dir_b / "pipeline.log").read_text(encoding="utf-8")

    assert "Message for run A" in log_a
    assert "Message for run B" not in log_a

    assert "Message for run B" in log_b
    assert "Message for run A" not in log_b

    # Cleanup run A
    remove_run_logging(dir_a, "pipeline.log")
    remaining_after_a = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(remaining_after_a) == len(file_handlers_both) - 1

    # Cleanup run B
    remove_run_logging(dir_b, "pipeline.log")


def test_console_handler_never_duplicated(tmp_path):
    """Repeated setup_logging calls do not duplicate console handlers."""
    logger = logging.getLogger("demandify")
    setup_logging()
    setup_logging(run_dir=tmp_path / "run_1")
    setup_logging(run_dir=tmp_path / "run_2")

    console_handlers = [h for h in logger.handlers if isinstance(h, TqdmLoggingHandler)]
    assert len(console_handlers) == 1

    # Clean up file handlers
    remove_run_logging(tmp_path / "run_1")
    remove_run_logging(tmp_path / "run_2")
