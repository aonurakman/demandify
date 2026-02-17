"""Tests for offline dataset discovery and artifact copy helpers."""

import json
from pathlib import Path

from demandify import offline_data


def _write_dataset(root: Path, name: str, bbox: dict):
    ds = root / name
    (ds / "data").mkdir(parents=True, exist_ok=True)
    (ds / "sumo").mkdir(parents=True, exist_ok=True)
    (ds / "plots").mkdir(parents=True, exist_ok=True)
    (ds / "data" / "traffic_data_raw.csv").write_text("segment_id,current_speed\ns1,20\n", encoding="utf-8")
    (ds / "data" / "observed_edges.csv").write_text(
        "edge_id,segment_id,current_speed,freeflow_speed,match_confidence\ne1,s1,20,40,1.0\n",
        encoding="utf-8",
    )
    (ds / "sumo" / "network.net.xml").write_text("<net/>", encoding="utf-8")
    (ds / "plots" / "network.png").write_bytes(b"png")
    (ds / "dataset_meta.json").write_text(
        json.dumps(
            {
                "dataset_name": name,
                "created_at": "2026-02-17T12:00:00",
                "bbox": bbox,
                "quality": {"label": "good", "score": 72},
                "provider": {"provider": "tomtom"},
            }
        ),
        encoding="utf-8",
    )
    return ds


def test_catalog_and_resolve(monkeypatch, tmp_path):
    generated_root = tmp_path / "generated"
    packaged_root = tmp_path / "packaged"
    generated_root.mkdir()
    packaged_root.mkdir()

    _write_dataset(
        generated_root,
        "gen_city_v1",
        {"west": 2.0, "south": 48.0, "east": 2.1, "north": 48.1},
    )
    _write_dataset(
        packaged_root,
        "pkg_city_v1",
        {"west": 20.0, "south": 50.0, "east": 20.1, "north": 50.1},
    )

    monkeypatch.setattr(offline_data, "GENERATED_DATASETS_ROOT", generated_root)
    monkeypatch.setattr(offline_data, "_packaged_datasets_root", lambda: packaged_root)

    catalog = offline_data.get_offline_dataset_catalog(include_generated=True, include_packaged=True)
    assert len(catalog) == 2
    names = {item["name"] for item in catalog}
    assert {"gen_city_v1", "pkg_city_v1"} == names

    resolved = offline_data.resolve_offline_dataset("pkg_city_v1")
    assert resolved.dataset_id == "packaged:pkg_city_v1"
    assert resolved.bbox["north"] == 50.1


def test_copy_offline_dataset_to_output(monkeypatch, tmp_path):
    packaged_root = tmp_path / "packaged"
    packaged_root.mkdir()
    _write_dataset(
        packaged_root,
        "krakow_v1",
        {"west": 20.0174, "south": 50.0702, "east": 20.0566, "north": 50.0875},
    )

    monkeypatch.setattr(offline_data, "GENERATED_DATASETS_ROOT", tmp_path / "generated_empty")
    monkeypatch.setattr(offline_data, "_packaged_datasets_root", lambda: packaged_root)

    resolved = offline_data.resolve_offline_dataset("krakow_v1")
    output_dir = tmp_path / "run_output"
    copied = offline_data.copy_offline_dataset_to_output(resolved, output_dir)

    assert (output_dir / "data" / "traffic_data_raw.csv").exists()
    assert (output_dir / "data" / "observed_edges.csv").exists()
    assert (output_dir / "sumo" / "network.net.xml").exists()
    assert (output_dir / "data" / "imported_dataset_meta.json").exists()
    assert "data/traffic_data_raw.csv" in copied


def test_normalize_offline_dataset_name_validation():
    assert offline_data.normalize_offline_dataset_name("krakow_v2") == "krakow_v2"
    assert offline_data.normalize_offline_dataset_name("  den_haag_v2  ") == "den_haag_v2"

    try:
        offline_data.normalize_offline_dataset_name("")
        assert False, "Expected ValueError for empty name"
    except ValueError:
        pass

    try:
        offline_data.normalize_offline_dataset_name("bad/name")
        assert False, "Expected ValueError for invalid characters"
    except ValueError:
        pass


def test_offline_dataset_name_exists(monkeypatch, tmp_path):
    generated_root = tmp_path / "generated"
    packaged_root = tmp_path / "packaged"
    generated_root.mkdir()
    packaged_root.mkdir()

    _write_dataset(
        generated_root,
        "gen_city_v1",
        {"west": 2.0, "south": 48.0, "east": 2.1, "north": 48.1},
    )
    _write_dataset(
        packaged_root,
        "pkg_city_v1",
        {"west": 20.0, "south": 50.0, "east": 20.1, "north": 50.1},
    )

    monkeypatch.setattr(offline_data, "GENERATED_DATASETS_ROOT", generated_root)
    monkeypatch.setattr(offline_data, "_packaged_datasets_root", lambda: packaged_root)

    assert offline_data.offline_dataset_name_exists("gen_city_v1") is True
    assert offline_data.offline_dataset_name_exists("pkg_city_v1") is True
    assert offline_data.offline_dataset_name_exists("missing_city_v1") is False


def test_get_writable_offline_datasets_root_falls_back_to_generated(monkeypatch, tmp_path):
    packaged_root = tmp_path / "packaged"
    generated_root = tmp_path / "generated"

    monkeypatch.setattr(offline_data, "PACKAGED_DATASETS_ROOT", packaged_root)
    monkeypatch.setattr(offline_data, "GENERATED_DATASETS_ROOT", generated_root)
    monkeypatch.setattr(
        offline_data,
        "_is_directory_writable",
        lambda p: p == generated_root,
    )

    selected = offline_data.get_writable_offline_datasets_root()
    assert selected == generated_root
