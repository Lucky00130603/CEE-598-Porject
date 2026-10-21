"""Generate a small synthetic table for testing the training pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.rows < 40:
        raise ValueError("rows must be at least 40")

    rng = np.random.default_rng(args.seed)
    station_number = rng.integers(0, 8, size=args.rows)
    p_3h = rng.gamma(shape=0.7, scale=0.25, size=args.rows)
    dry = rng.random(args.rows) < 0.65
    p_3h[dry] = 0.0

    fraction_2h = rng.uniform(0.60, 0.95, size=args.rows)
    fraction_1h = rng.uniform(0.45, 0.90, size=args.rows)
    fraction_30min = rng.uniform(0.45, 0.90, size=args.rows)
    fraction_15min = rng.uniform(0.45, 0.90, size=args.rows)
    p_2h = p_3h * fraction_2h
    p_1h = p_2h * fraction_1h
    p_30min = p_1h * fraction_30min
    p_15min = p_30min * fraction_15min

    latitude = 37.0 + station_number * 0.55
    longitude = -91.0 + station_number * 0.45
    elevation = 120.0 + station_number * 18.0
    daily_precip = p_3h + rng.gamma(0.6, 0.08, size=args.rows)
    tmax = rng.normal(72.0, 14.0, size=args.rows)
    tmin = tmax - rng.uniform(8.0, 24.0, size=args.rows)

    frame = pd.DataFrame(
        {
            "station_id": [f"DEMO-{number:02d}" for number in station_number],
            "timestamp": pd.date_range(
                "2000-01-01", periods=args.rows, freq="3h"
            ).astype(str),
            "p_3h": p_3h,
            "p_2h": p_2h,
            "p_1h": p_1h,
            "p_30min": p_30min,
            "p_15min": p_15min,
            "daily_precip": daily_precip,
            "tmax": tmax,
            "tmin": tmin,
            "latitude": latitude,
            "longitude": longitude,
            "elevation": elevation,
            "neighbor_idw_mean": np.maximum(
                0.0, p_3h + rng.normal(0.0, 0.03, size=args.rows)
            ),
            "neighbor_max": p_3h + rng.uniform(0.0, 0.08, size=args.rows),
            "neighbor_std": rng.uniform(0.0, 0.05, size=args.rows),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {args.output}")


if __name__ == "__main__":
    main()
