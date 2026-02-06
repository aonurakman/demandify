"""Tests for the CLI restart prompt logic."""

from unittest.mock import patch
from demandify.cli import _prompt_restart


def test_prompt_restart_yes():
    with patch("builtins.input", return_value="y"):
        assert _prompt_restart() is True


def test_prompt_restart_yes_uppercase():
    with patch("builtins.input", return_value="Y"):
        assert _prompt_restart() is True


def test_prompt_restart_no():
    with patch("builtins.input", return_value="n"):
        assert _prompt_restart() is False


def test_prompt_restart_empty():
    with patch("builtins.input", return_value=""):
        assert _prompt_restart() is False


def test_prompt_restart_eof():
    with patch("builtins.input", side_effect=EOFError):
        assert _prompt_restart() is False
