"""Tests for network-plot observed-edge highlighting."""

from pathlib import Path

from shapely.geometry import LineString

from demandify.utils import visualization


class _FakeNet:
    def __init__(self):
        self.edges = ["e1", "e2"]
        self.edge_geometries = {
            "e1": LineString([(0.0, 0.0), (1.0, 1.0)]),
            "e2": LineString([(1.0, 0.0), (2.0, 1.0)]),
        }

    def get_network_boundary(self):
        return (0.0, 0.0, 2.0, 1.0)


class _FakeAxes:
    def __init__(self):
        self.plot_calls = []
        self.text_calls = []
        self.aspect = None
        self.axis_value = None
        self.transAxes = object()

    def plot(self, _x, _y, **kwargs):
        self.plot_calls.append(kwargs)

    def set_aspect(self, value):
        self.aspect = value

    def axis(self, value):
        self.axis_value = value

    def text(self, *_args, **kwargs):
        self.text_calls.append(kwargs)


class _FakeFigure:
    def __init__(self):
        self.saved = None

    def savefig(self, output_file, **kwargs):
        self.saved = (output_file, kwargs)


def test_plot_network_geometry_overlays_observed_edges(monkeypatch):
    fake_ax = _FakeAxes()
    fake_fig = _FakeFigure()

    monkeypatch.setattr(visualization, "SUMONetwork", lambda _network_file: _FakeNet())
    monkeypatch.setattr(visualization.plt, "subplots", lambda **_kwargs: (fake_fig, fake_ax))
    monkeypatch.setattr(visualization.plt, "close", lambda _fig: None)

    visualization.plot_network_geometry(
        Path("/tmp/network.net.xml"),
        Path("/tmp/network.png"),
        observed_edge_ids={"e2"},
    )

    colors = [call.get("color") for call in fake_ax.plot_calls]
    assert "#333333" in colors
    assert "#e53935" in colors
    assert fake_fig.saved is not None


def test_plot_edge_speed_heatmap_saves_png(monkeypatch, tmp_path):
    monkeypatch.setattr(visualization, "SUMONetwork", lambda _network_file: _FakeNet())

    output_file = tmp_path / "network_observed_speed_heatmap.png"
    visualization.plot_edge_speed_heatmap(
        Path("/tmp/network.net.xml"),
        output_file,
        edge_speeds={"e1": 22.5, "e2": 41.0},
        title="Observed Edge Speeds",
        vmin=0.0,
        vmax=50.0,
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_od_pair_labels_saves_png(monkeypatch, tmp_path):
    monkeypatch.setattr(visualization, "SUMONetwork", lambda _network_file: _FakeNet())

    output_file = tmp_path / "network_selected_od_pairs.png"
    visualization.plot_od_pair_labels(
        Path("/tmp/network.net.xml"),
        output_file,
        od_pairs=[("e1", "e2"), ("e2", "e1")],
        title="Selected ODs (2)",
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0
