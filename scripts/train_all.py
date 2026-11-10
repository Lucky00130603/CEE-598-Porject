"""Run all four temporal-downscaling training tasks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


TASK_FILES = {
    "3h_to_2h": "pairs_3h_to_2h.csv",
    "2h_to_1h": "pairs_2h_to_1h.csv",
    "1h_to_30min": "pairs_1h_to_30min.csv",
    "30min_to_15min": "pairs_30min_to_15min.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data", type=Path, help="Combined training table")
    data_group.add_argument("--data-dir", type=Path, help="Directory of paired task CSVs")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument("--split", choices=("station", "random"), default="random")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for task, filename in TASK_FILES.items():
        data_path = args.data if args.data is not None else args.data_dir / filename
        command = [
            sys.executable,
            "-m",
            "src.train",
            "--data",
            str(data_path),
            "--task",
            task,
            "--output",
            str(args.output_root / task),
            "--split",
            args.split,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch-size",
            str(args.batch_size),
            "--learning-rate",
            str(args.learning_rate),
            "--seed",
            str(args.seed),
        ]
        if args.spatial:
            command.append("--spatial")
        print(f"Running {task}...")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
