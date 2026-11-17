"""Shared pytest fixtures for test isolation."""

import pytest


@pytest.fixture(autouse=True)
def _isolate_demandify_config(tmp_path, monkeypatch):
    """Isolate config/cache paths per test so tests never touch ~/.demandify."""
    import demandify.config as config_module

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("CACHE_DIR", str(fake_home / ".demandify" / "cache"))
    monkeypatch.setattr(config_module, "_config_instance", None)
    yield
    monkeypatch.setattr(config_module, "_config_instance", None)
