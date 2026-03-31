# Offline Dataset Bundles

This directory is for curated offline datasets that should be shipped in Git and pip releases.

Recommended structure:

```
offline_datasets/
  <dataset_id>/
    dataset_meta.json
    data/
      traffic_data_raw.csv
      observed_edges.csv
    sumo/
      network.net.xml
```

Notes:

- Do not add raw `.osm` files here; they are excluded from packaging.
- `data/observed_edges.csv` should ideally include `sumo_freeflow_speed_kmh`; if it does not, the runtime pipeline will enrich it from the packaged SUMO network before calibration.
- Runtime-generated datasets are created in `demandify_datasets/` and are ignored by Git.
- Promote a runtime dataset by copying only the required files into this directory.
