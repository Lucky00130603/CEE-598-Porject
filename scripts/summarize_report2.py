"""Create Report II summary tables and figures from four training outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TASKS = ["3h_to_2h", "2h_to_1h", "1h_to_30min", "30min_to_15min"]
LABELS = {
    "3h_to_2h": "3 h to 2 h",
    "2h_to_1h": "2 h to 1 h",
    "1h_to_30min": "1 h to 30 min",
    "30min_to_15min": "30 min to 15 min",
}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r": float(np.corrcoef(observed, predicted)[0, 1]),
        "r2": float(r2_score(observed, predicted)),
        "negative_fraction": float(np.mean(predicted < 0)),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    dataset_rows = []
    quality_rows = []
    model_rows = []
    threshold_rows = []
    negative_rows = []
    histories: dict[str, pd.DataFrame] = {}
    predictions: dict[str, pd.DataFrame] = {}

    for task in TASKS:
        task_dir = args.input_root / task
        summary = load_json(task_dir / "data_summary.json")
        result = load_json(task_dir / "metrics.json")
        prediction = pd.read_csv(task_dir / "predictions.csv")
        history = pd.read_csv(task_dir / "history.csv")
        histories[task] = history
        predictions[task] = prediction

        clean = summary["cleaning"]
        target = summary["target_statistics"]
        dataset_rows.append(
            {
                "task": LABELS[task],
                "raw_rows": clean["raw_rows"],
                "retained_rows": clean["final_rows"],
                "removed_rows": clean["removed_union_rows"],
                "retained_fraction": clean["final_rows"] / clean["raw_rows"],
                "stations": summary["station_count"],
                "start": summary["timestamp_start"][:10],
                "end": summary["timestamp_end"][:10],
                "zero_fraction": target["zero_fraction"],
                "target_mean": target["mean"],
                "target_p99": target["p99"],
                "target_max": target["maximum"],
                "train_rows": summary["training_samples"],
                "validation_rows": summary["validation_samples"],
                "test_rows": summary["testing_samples"],
            }
        )
        quality_rows.append(
            {
                "task": LABELS[task],
                "input_gt_daily": clean["input_exceeds_daily_rows"],
                "target_gt_daily": clean["target_exceeds_daily_rows"],
                "temperature_inversion": clean["temperature_inversion_rows"],
                "removed_union": clean["removed_union_rows"],
            }
        )

        for name, key in (
            ("ANN", "ann"),
            ("Persistence", "persistence_baseline"),
            ("Duration scaling", "duration_scaling_baseline"),
        ):
            values = result[key]
            model_rows.append(
                {
                    "task": LABELS[task],
                    "model": name,
                    "n": values["n"],
                    "mae": values["mae"],
                    "rmse": values["rmse"],
                    "r": values["r"],
                    "r2": values["r2"],
                    "negative_fraction": values["negative_prediction_fraction"],
                }
            )

        clipped = np.maximum(prediction["predicted"].to_numpy(), 0.0)
        observed = prediction["observed"].to_numpy()
        clipped_metrics = metrics(observed, clipped)
        model_rows.append(
            {
                "task": LABELS[task],
                "model": "ANN clipped at zero",
                **clipped_metrics,
            }
        )
        raw_predicted = prediction["predicted"].to_numpy()
        negative = raw_predicted[raw_predicted < 0]
        negative_rows.append(
            {
                "task": LABELS[task],
                "negative_count": int(len(negative)),
                "negative_fraction": float(np.mean(raw_predicted < 0)),
                "negative_min": float(negative.min()) if len(negative) else 0.0,
                "negative_mean": float(negative.mean()) if len(negative) else 0.0,
                "negative_with_zero_observed_fraction": float(
                    np.mean(observed[raw_predicted < 0] == 0)
                )
                if len(negative)
                else 0.0,
            }
        )

        for threshold_key, values in result["ann_by_threshold"].items():
            threshold_rows.append(
                {
                    "task": LABELS[task],
                    "threshold": threshold_key,
                    "n": values["n"],
                    "mae": values["mae"],
                    "rmse": values["rmse"],
                    "r": values["r"],
                    "r2": values["r2"],
                }
            )

    datasets = pd.DataFrame(dataset_rows)
    quality = pd.DataFrame(quality_rows)
    models = pd.DataFrame(model_rows)
    thresholds = pd.DataFrame(threshold_rows)
    negatives = pd.DataFrame(negative_rows)
    datasets.to_csv(args.output / "dataset_summary.csv", index=False)
    quality.to_csv(args.output / "quality_issues.csv", index=False)
    models.to_csv(args.output / "model_metrics.csv", index=False)
    thresholds.to_csv(args.output / "threshold_metrics.csv", index=False)
    negatives.to_csv(args.output / "negative_diagnostics.csv", index=False)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.5), constrained_layout=True)
    for ax, task, color in zip(axes.flat, TASKS, COLORS):
        history = histories[task]
        ax.plot(history["epoch"], history["training_loss"], color=color, label="Training")
        ax.plot(
            history["epoch"],
            history["validation_loss"],
            color="#333333",
            linestyle="--",
            label="Validation",
        )
        ax.set_title(LABELS[task])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Standardized MSE")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.savefig(args.output / "training_curves.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.1), constrained_layout=True)
    for ax, task in zip(axes.flat, TASKS):
        prediction = predictions[task]
        observed = prediction["observed"].to_numpy()
        predicted = np.maximum(prediction["predicted"].to_numpy(), 0.0)
        limit = max(float(np.quantile(observed, 0.997)), float(np.quantile(predicted, 0.997)))
        limit = max(limit, 0.1)
        sample = prediction.sample(min(len(prediction), 60_000), random_state=42)
        sample_predicted = np.maximum(sample["predicted"].to_numpy(), 0.0)
        ax.hexbin(
            sample["observed"],
            sample_predicted,
            gridsize=55,
            extent=(0, limit, 0, limit),
            mincnt=1,
            cmap="viridis",
            bins="log",
        )
        ax.plot([0, limit], [0, limit], color="#D55E00", linewidth=1, linestyle="--")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(LABELS[task])
        ax.set_xlabel("Observed precipitation (in)")
        ax.set_ylabel("ANN prediction (in)")
    fig.savefig(args.output / "observed_vs_predicted.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8), constrained_layout=True)
    x = np.arange(len(TASKS))
    width = 0.24
    for offset, model_name, color in (
        (-width, "Persistence", "#999999"),
        (0.0, "Duration scaling", "#E69F00"),
        (width, "ANN", "#0072B2"),
    ):
        values = [
            models.loc[(models.task == LABELS[t]) & (models.model == model_name), "r2"].iloc[0]
            for t in TASKS
        ]
        axes[0].bar(x + offset, values, width, label=model_name, color=color)
    axes[0].set_xticks(x, [LABELS[t] for t in TASKS], rotation=25, ha="right")
    axes[0].set_ylabel("Test $R^2$")
    axes[0].set_ylim(0.6, 1.0)
    axes[0].set_title("Model comparison")
    axes[0].legend(frameon=False, loc="lower left")
    axes[0].grid(axis="y", alpha=0.25)

    threshold_labels = ["All", ">0", ">=0.1", ">=0.2", ">=0.3"]
    threshold_keys = [None, "target_gt_0", "target_ge_0.1", "target_ge_0.2", "target_ge_0.3"]
    for task, color in zip(TASKS, COLORS):
        values = []
        for key in threshold_keys:
            if key is None:
                values.append(
                    models.loc[(models.task == LABELS[task]) & (models.model == "ANN"), "r2"].iloc[0]
                )
            else:
                values.append(
                    thresholds.loc[
                        (thresholds.task == LABELS[task]) & (thresholds.threshold == key), "r2"
                    ].iloc[0]
                )
        axes[1].plot(threshold_labels, values, marker="o", label=LABELS[task], color=color)
    axes[1].set_ylabel("ANN $R^2$")
    axes[1].set_title("Performance for larger events")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].grid(alpha=0.25)

    raw_negative = negatives["negative_fraction"].to_numpy() * 100
    axes[2].bar(x, raw_negative, color=COLORS)
    axes[2].set_xticks(x, [LABELS[t] for t in TASKS], rotation=25, ha="right")
    axes[2].set_ylabel("Negative predictions (%)")
    axes[2].set_title("Physical feasibility issue")
    axes[2].grid(axis="y", alpha=0.25)
    fig.savefig(args.output / "performance_summary.png", dpi=300, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    axes[0].bar(x - 0.18, datasets["raw_rows"] / 1000, 0.36, label="Raw", color="#999999")
    axes[0].bar(
        x + 0.18,
        datasets["retained_rows"] / 1000,
        0.36,
        label="Retained",
        color="#0072B2",
    )
    axes[0].set_xticks(x, datasets["task"], rotation=25, ha="right")
    axes[0].set_ylabel("Observations (thousands)")
    axes[0].set_title("Paired data retained after QC")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, datasets["zero_fraction"] * 100, color=COLORS)
    axes[1].set_xticks(x, datasets["task"], rotation=25, ha="right")
    axes[1].set_ylabel("Zero target precipitation (%)")
    axes[1].set_title("Class imbalance across tasks")
    axes[1].grid(axis="y", alpha=0.25)
    fig.savefig(args.output / "data_quality_summary.png", dpi=300, facecolor="white")
    plt.close(fig)

    print(f"Wrote report tables and figures to {args.output}")


if __name__ == "__main__":
    main()
