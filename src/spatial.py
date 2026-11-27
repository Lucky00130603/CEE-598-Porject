"""Build NOAA spatial predictors using the five geographically nearest gauges."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .train import TASKS


def enrich(frame, metadata):
    fields = ["latitude", "longitude", "elevation"]
    if metadata["STATION"].duplicated().any():
        raise ValueError("Station metadata contains duplicate IDs")
    coordinates = metadata.set_index("STATION")[fields].apply(pd.to_numeric, errors="raise")
    missing = sorted(set(frame.STATION) - set(coordinates.dropna().index))
    if missing:
        raise ValueError(f"Missing latitude, longitude or elevation for {missing}")
    if frame.duplicated(["STATION", "timestamp"]).any():
        raise ValueError("NOAA pairs contain duplicate station/timestamps")
    stations = sorted(frame.STATION.unique())
    if len(stations) < 6:
        raise ValueError("Five neighbors require at least six stations")
    origin = coordinates.loc[stations]
    radians = np.deg2rad(origin[["latitude", "longitude"]].to_numpy())
    lat, lon = radians.T
    deltas = np.sin((lat[:, None] - lat) / 2) ** 2
    deltas += np.cos(lat[:, None]) * np.cos(lat) * np.sin((lon[:, None] - lon) / 2) ** 2
    distance = 2 * np.arcsin(np.sqrt(np.clip(deltas, 0, 1)))
    np.fill_diagonal(distance, np.inf)
    by_station = {key: rows.set_index("timestamp") for key, rows in frame.groupby("STATION")}
    parts = []
    for index, station in enumerate(stations):
        neighbors = np.argsort(distance[index])[:5]
        rows = by_station[station].copy()
        values = np.column_stack([by_station[stations[j]].P_in.reindex(rows.index) for j in neighbors])
        complete = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
        rows = rows.loc[complete].copy()
        values = values[complete]
        weights = 1 / np.maximum(distance[index, neighbors], 1e-10)
        rows[fields] = np.tile(origin.loc[station].to_numpy(), (len(rows), 1))
        rows["neighbor_idw_mean"] = np.average(values, weights=weights, axis=1)
        rows["neighbor_max"] = values.max(axis=1)
        rows["neighbor_std"] = values.std(axis=1)
        parts.append(rows.reset_index())
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--stations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    metadata = pd.read_csv(args.stations, dtype={"STATION": str})
    args.out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        name = f"pairs_{task}.csv"
        frame = pd.read_csv(args.pairs / name, dtype={"STATION": str})
        joined = enrich(frame, metadata)
        if joined.STATION.nunique() < 4:
            raise ValueError(f"{task}: too few stations with all five simultaneous neighbors")
        joined.to_csv(args.out / name, index=False)
        print(f"{task}: {len(joined):,}/{len(frame):,} rows with all five neighbors", flush=True)


if __name__ == "__main__":
    main()
