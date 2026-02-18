"""Tests for TomTom tile ingestion diagnostics."""

import asyncio
import logging

from demandify.providers.tomtom import HAS_MVT, TomTomProvider


def _empty_feature_drop_reasons():
    return {
        "missing_geometry": 0,
        "unsupported_geometry_type": 0,
        "empty_geometry_lines": 0,
        "short_geometry_line": 0,
        "invalid_geometry_point": 0,
        "missing_speed_property": 0,
    }


def test_tile_mode_is_disabled_by_default_and_reenable_via_env(monkeypatch):
    monkeypatch.delenv("DEMANDIFY_ENABLE_TOMTOM_TILES", raising=False)
    provider_default = TomTomProvider("dummy")
    try:
        assert provider_default.use_tiles is False
    finally:
        asyncio.run(provider_default.close())

    monkeypatch.setenv("DEMANDIFY_ENABLE_TOMTOM_TILES", "1")
    provider_env_on = TomTomProvider("dummy")
    try:
        assert provider_env_on.use_tiles is HAS_MVT
    finally:
        asyncio.run(provider_env_on.close())


def test_decode_tile_counts_missing_speed_drop(monkeypatch):
    provider = TomTomProvider("dummy")

    def fake_decode(_tile_bytes):
        return {
            "flow": {
                "extent": 4096,
                "features": [
                    {
                        "properties": {"roadClass": "FRC2"},
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [10, 10]]},
                    }
                ],
            }
        }

    monkeypatch.setattr("demandify.providers.tomtom.mvt_decode", fake_decode)

    try:
        segments, diag = provider._decode_tile(b"dummy", 2180, 1357)
        assert segments == []
        assert diag["features_total"] == 1
        assert diag["feature_drop_reasons"]["missing_speed_property"] == 1
        assert diag["segments_decoded"] == 0
    finally:
        asyncio.run(provider.close())


def test_decode_tile_accepts_traffic_level_speed(monkeypatch):
    provider = TomTomProvider("dummy")

    def fake_decode(_tile_bytes):
        return {
            "Traffic flow": {
                "extent": 4096,
                "features": [
                    {
                        "properties": {
                            "road_type": "Major road",
                            "traffic_level": 37.0,
                        },
                        "geometry": {"type": "LineString", "coordinates": [[0, 0], [16, 16]]},
                    }
                ],
            }
        }

    monkeypatch.setattr("demandify.providers.tomtom.mvt_decode", fake_decode)

    try:
        segments, diag = provider._decode_tile(b"dummy", 2275, 1387)
        assert len(segments) == 1
        assert segments[0]["current_speed"] == 37.0
        assert segments[0]["freeflow_speed"] == 37.0
        assert diag["feature_drop_reasons"]["missing_speed_property"] == 0
        assert diag["segments_decoded"] == 1
    finally:
        asyncio.run(provider.close())


def test_fetch_via_tiles_logs_drop_reason_summary(monkeypatch, caplog):
    provider = TomTomProvider("dummy")
    provider.use_tiles = True

    monkeypatch.setattr(provider, "_bbox_to_tiles", lambda _bbox: [(2180, 1357), (2180, 1358)])

    async def fake_fetch_tile(_x, y):
        if y == 1357:
            return b"tile-bytes", "ok"
        return None, "http_403"

    monkeypatch.setattr(provider, "_fetch_tile", fake_fetch_tile)

    def fake_decode_tile(_tile_bytes, _x, _y):
        drops = _empty_feature_drop_reasons()
        drops["missing_speed_property"] = 3
        return [], {
            "decode_error": 0,
            "layers_total": 1,
            "features_total": 3,
            "feature_drop_reasons": drops,
            "segments_decoded": 0,
        }

    monkeypatch.setattr(provider, "_decode_tile", fake_decode_tile)
    caplog.set_level(logging.WARNING, logger="demandify.providers.tomtom")

    try:
        df = asyncio.run(provider._fetch_via_tiles((20.0, 50.0, 20.1, 50.1)))
        assert df.empty

        summary_logs = [
            rec.getMessage() for rec in caplog.records if "Tile mode yielded 0 usable segments" in rec.getMessage()
        ]
        assert len(summary_logs) == 1
        assert "http_403=1" in summary_logs[0]
        assert "missing_speed_property=3" in summary_logs[0]
    finally:
        asyncio.run(provider.close())
