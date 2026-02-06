import shutil

from demandify.utils.validation import check_sumo_availability


def test_check_sumo_availability_reports_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    ok, msg = check_sumo_availability()
    assert ok is False
    assert "netconvert" in msg
    assert "duarouter" in msg
    assert "sumo" in msg


def test_check_sumo_availability_ok(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/fake")
    ok, msg = check_sumo_availability()
    assert ok is True
    assert "All SUMO tools found" in msg

