"""Run the NOAA baseline and NARCCAP assessment with browser-readable progress."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import torch

from .climate import DURATIONS, MODELS, PERIODS, climate_run, read_values
from .train import TASKS


class Status:
    def __init__(self, root):
        self.path = root / "status.json"
        self.events = []
        self.metrics = []
        self.files = []
        self.update("Run started")

    def update(self, message=None, state="Running"):
        if message:
            self.events.append(message)
            print(message, flush=True)
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps({
            "state": state, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "events": self.events, "metrics": self.metrics, "files": self.files,
        }), encoding="utf-8")
        temp.replace(self.path)

    def publish(self, path):
        link = path.relative_to(self.path.parent).as_posix()
        if link not in self.files:
            self.files.append(link)


def extract_pairs(archive, folder):
    folder.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as zipped:
        for task in TASKS:
            name = f"pairs_{task}.csv"
            members = [info for info in zipped.infolist() if Path(info.filename).name == name]
            if len(members) != 1:
                raise ValueError(f"{archive}: expected one {name}")
            with zipped.open(members[0]) as source, (folder / name).open("wb") as dest:
                while block := source.read(1024 * 1024):
                    dest.write(block)


def train(args, status):
    root = args.run_dir / "training"
    for task in TASKS:
        path = args.pairs / f"pairs_{task}.csv"
        if not path.is_file():
            raise ValueError(f"Missing training table {path}")
        output = root / task
        status.update(f"Training NOAA {task}")
        env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "MKL_NUM_THREADS": str(args.threads)}
        with subprocess.Popen([
            sys.executable, "-m", "src.train", "--task", task, "--data", str(path),
            "--output", str(output), "--epochs", str(args.epochs), "--patience", "8",
            "--batch-size", "4096", "--learning-rate", "0.003", "--split", args.split,
            "--progress",
        ], stdout=subprocess.PIPE, text=True, env=env) as process:
            try:
                for line in process.stdout:
                    if line.startswith("Epoch "):
                        status.update(f"{task} {line.strip()}")
                if process.wait() != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)
            except KeyboardInterrupt:
                process.terminate()
                raise
        add_metrics(task, output, status)
        status.update(f"Finished NOAA {task}")
    subprocess.run([
        sys.executable, "scripts/summarize_report2.py",
        "--input-root", str(root), "--output", str(args.run_dir / "evaluation"),
    ], check=True)
    for path in sorted((args.run_dir / "evaluation").iterdir()):
        if path.suffix in {".png", ".csv"}:
            status.publish(path)
    status.update("NOAA figures and tables ready")


def add_metrics(task, output, status):
    metrics = json.loads((output / "metrics.json").read_text())
    summary = json.loads((output / "data_summary.json").read_text())
    model = metrics["ann"]
    status.metrics.append([
        task, summary["cleaning"]["final_rows"], model["n"],
        *(f"{model[k]:.4f}" for k in ("mae", "rmse", "r2")),
        f'{metrics["duration_scaling_baseline"]["r2"]:.4f}',
    ])


def plot_idf(paths, root, latitude, longitude):
    metadata = []
    rows = []
    figure, axes = plt.subplots(1, len(paths), figsize=(6.3 * len(paths), 4.1), squeeze=False)
    try:
        for ax, path in zip(axes[0], paths):
            with nc.Dataset(path) as ds:
                lat = read_values(ds["lat"])
                lon = (read_values(ds["lon"]) + 180) % 360 - 180
                distance = (lat - latitude) ** 2 + ((lon - longitude) * np.cos(np.deg2rad(latitude))) ** 2
                position = np.unravel_index(np.nanargmin(distance), lat.shape)
                maxima = read_values(ds["annual_max_depth"], (slice(None), slice(None), *position))
                model, period = ds.climate_model, ds.period
                years = np.asarray(ds["year"][:])
                metadata.append((model, period, *years[[0, -1]], lat[position], lon[position]))
                for minutes, depths in zip(DURATIONS, maxima.T):
                    intensities = np.sort(depths[np.isfinite(depths)])[::-1] * 60 / minutes
                    n = len(intensities)
                    return_period = (n + 1) / np.arange(1, n + 1)
                    rows.extend((model, period, minutes, n, r, v) for r, v in zip(return_period, intensities))
                    if n:
                        ax.plot(return_period, intensities, marker=".", label=f"{minutes} min (n={n})")
                ax.set(xscale="log", xlabel="Empirical return period (years)", ylabel="Intensity (in/hour)",
                       title=f"{model} · {period}\n{lat[position]:.2f}°, {lon[position]:.2f}°")
                ax.grid(alpha=0.2)
                if ax.lines:
                    ax.legend(fontsize=8)
        figure.suptitle("Annual window maxima · empirical IDF")
        figure.tight_layout()
        figure.savefig(root / "empirical_idf.png", dpi=135)
    finally:
        plt.close(figure)
    with (root / "empirical_idf.csv").open("w", newline="", encoding="utf-8") as table:
        writer = csv.writer(table)
        writer.writerow(["model", "period", "duration_min", "valid_years", "return_period_years", "intensity_in_per_hour"])
        writer.writerows(rows)
    with (root / "idf_grid_cells.csv").open("w", newline="", encoding="utf-8") as table:
        writer = csv.writer(table)
        writer.writerow(["model", "period", "first_year", "last_year", "latitude", "longitude"])
        writer.writerows(metadata)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "predict", "all"])
    parser.add_argument("--run-dir", type=Path, default=Path("results/final"))
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/noaa_pairs"))
    parser.add_argument("--archive", type=Path, default=Path("data/noaa data.zip"))
    parser.add_argument("--climate-dir", type=Path, default=Path("data/raw/narccap"))
    parser.add_argument("--weights", type=Path, help="Folder with four trusted subfolders, each with model.pt")
    parser.add_argument("--model", choices=MODELS, default="ecp2")
    parser.add_argument("--period", choices=["historical", "future", "both"], default="both")
    parser.add_argument("--historical-years", nargs=2, type=int, default=PERIODS["historical"])
    parser.add_argument("--future-years", nargs=2, type=int, default=PERIODS["future"])
    parser.add_argument("--stamp", choices=["start", "center", "end"], help="Timestamp position if pr has no time bounds")
    parser.add_argument("--split", choices=["random", "station"], default="random")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-windows", action="store_true")
    parser.add_argument("--site", nargs=2, type=float, default=(40.11, -88.24), metavar=("LAT", "LON"))
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.threads) < 1:
        raise SystemExit("Epochs and threads must be positive")
    if args.stage != "predict" and args.weights:
        raise SystemExit("--weights applies to predict; all uses the checkpoints trained in that run")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    status = Status(args.run_dir)
    try:
        torch.set_num_threads(args.threads)
        weights = args.weights or args.run_dir / "training"
        if args.stage in {"train", "all"}:
            if any(not (args.pairs / f"pairs_{task}.csv").is_file() for task in TASKS):
                extract_pairs(args.archive, args.pairs)
            train(args, status)
        if args.stage == "predict":
            for task in TASKS:
                folder = weights / task
                if (folder / "metrics.json").exists() and (folder / "data_summary.json").exists():
                    add_metrics(task, folder, status)
            for path in sorted((args.run_dir / "evaluation").glob("*")):
                if path.suffix in {".png", ".csv"}:
                    status.publish(path)
        if args.stage in {"predict", "all"}:
            files = []
            periods = PERIODS if args.period == "both" else [args.period]
            for period in periods:
                years = getattr(args, f"{period}_years")
                if not 1900 <= years[0] <= years[1] <= 2200:
                    raise ValueError(f"Invalid {period} years: {years}")
                result = args.run_dir / args.model / period
                path = climate_run(args.climate_dir, args.model, period, years, args.stamp,
                                   weights, result, status.update, args.save_windows)
                files.append(path)
                status.publish(path)
                if args.save_windows:
                    status.publish(result / "window_depths.nc")
            plot_idf(files, args.run_dir, *args.site)
            for name in ("empirical_idf.csv", "idf_grid_cells.csv", "empirical_idf.png"):
                status.publish(args.run_dir / name)
        status.update("Requested stages completed", "Complete")
    except (Exception, KeyboardInterrupt) as exc:
        status.update(str(exc) or "Interrupted", "Failed")
        raise


if __name__ == "__main__":
    main()
