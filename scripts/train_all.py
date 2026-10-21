"""Run all four temporal-downscaling training tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASKS = ("3h_to_2h", "2h_to_1h", "1h_to_30min", "30min_to_15min")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--split", choices=("station", "random"), default="station")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for task in TASKS:
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--data",
            str(args.data),
            "--task",
            task,
            "--output",
            str(args.output_root / task),
            "--split",
            args.split,
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
        ]
        if args.spatial:
            command.append("--spatial")
        print(f"Running {task}...")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
