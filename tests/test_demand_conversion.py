import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from demandify.sumo.demand import DemandGenerator


def test_genome_to_csv_and_trips_xml(tmp_path):
    demand_gen = DemandGenerator(network=None, seed=123)

    od_pairs = [("edgeA", "edgeB"), ("edgeC", "edgeD")]
    departure_bins = [(0, 60), (60, 120)]
    genome = np.array([1, 0, 2, 1])  # 2 OD * 2 bins

    demand_csv = tmp_path / "demand.csv"
    trips_xml = tmp_path / "trips.xml"

    df = demand_gen.genome_to_demand_csv(genome, od_pairs, departure_bins, demand_csv)

    assert demand_csv.exists()
    assert list(df.columns) == ["ID", "origin link id", "destination link id", "departure timestep"]
    assert len(df) == int(genome.sum())
    assert df["departure timestep"].is_monotonic_increasing

    # Departures should be within [0, 120)
    assert (df["departure timestep"] >= 0).all()
    assert (df["departure timestep"] < 120).all()

    # Roundtrip: CSV -> trips.xml
    demand_gen.demand_csv_to_trips_xml(demand_csv, trips_xml)
    assert trips_xml.exists()

    tree = ET.parse(trips_xml)
    root = tree.getroot()
    trips = root.findall("trip")
    assert len(trips) == len(df)

    # Validate a few fields match the CSV
    df_re = pd.read_csv(demand_csv)
    sample = df_re.head(3).to_dict(orient="records")
    for row, trip in zip(sample, trips[:3]):
        assert trip.get("id") == row["ID"]
        assert int(float(trip.get("depart"))) == int(row["departure timestep"])
        assert trip.get("from") == row["origin link id"]
        assert trip.get("to") == row["destination link id"]

