"""Train and evaluate one temporal-downscaling ANN."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import PrecipitationDownscalingANN


TASKS = {
    "3h_to_2h": ("p_3h", "p_2h"),
    "2h_to_1h": ("p_2h", "p_1h"),
    "1h_to_30min": ("p_1h", "p_30min"),
    "30min_to_15min": ("p_30min", "p_15min"),
}

DAILY_FEATURES = ["daily_precip", "tmax", "tmin"]
PAIR_FEATURES = ["P_in", "P_d", "Tmax_d", "Tmin_d"]
SPATIAL_FEATURES = [
    "latitude",
    "longitude",
    "elevation",
    "neighbor_idw_mean",
    "neighbor_max",
    "neighbor_std",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--spatial", action="store_true")
    parser.add_argument(
        "--split",
        choices=("station", "random"),
        default="random",
        help="Use random splitting for baseline runs or station holdout for spatial generalization.",
    )
    parser.add_argument(
        "--keep-physical-inconsistencies",
        action="store_true",
        help="Keep records that violate precipitation or temperature consistency checks.",
    )
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def detect_schema(
    frame: pd.DataFrame, task: str, spatial: bool
) -> tuple[list[str], str, str | None, str, str]:
    """Return features, target, station column, input column, and schema name."""
    pair_required = {*PAIR_FEATURES, "P_target"}
    if pair_required.issubset(frame.columns):
        if spatial:
            raise ValueError("Pair files do not yet contain spatial predictor columns")
        station = "STATION" if "STATION" in frame.columns else None
        return PAIR_FEATURES.copy(), "P_target", station, "P_in", "paired"

    source, target = TASKS[task]
    features = [source, *DAILY_FEATURES]
    if spatial:
        features.extend(SPATIAL_FEATURES)
    station = "station_id" if "station_id" in frame.columns else None
    return features, target, station, source, "combined"


def cleaning_diagnostics(frame: pd.DataFrame, schema: str) -> dict[str, int]:
    diagnostics = {
        "raw_rows": int(len(frame)),
        "missing_required_rows": 0,
        "negative_precipitation_rows": 0,
        "target_exceeds_input_rows": 0,
        "input_exceeds_daily_rows": 0,
        "target_exceeds_daily_rows": 0,
        "temperature_inversion_rows": 0,
        "removed_union_rows": 0,
    }
    if schema != "paired":
        return diagnostics

    diagnostics.update(
        {
            "negative_precipitation_rows": int(
                ((frame[["P_in", "P_target", "P_d"]] < 0).any(axis=1)).sum()
            ),
            "target_exceeds_input_rows": int(
                (frame["P_target"] > frame["P_in"] + 1e-10).sum()
            ),
            "input_exceeds_daily_rows": int(
                (frame["P_in"] > frame["P_d"] + 1e-10).sum()
            ),
            "target_exceeds_daily_rows": int(
                (frame["P_target"] > frame["P_d"] + 1e-10).sum()
            ),
            "temperature_inversion_rows": int(
                (frame["Tmin_d"] > frame["Tmax_d"]).sum()
            ),
        }
    )
    return diagnostics


def physical_consistency_mask(frame: pd.DataFrame, schema: str) -> pd.Series:
    if schema != "paired":
        return pd.Series(True, index=frame.index)
    return (
        (frame[["P_in", "P_target", "P_d"]] >= 0).all(axis=1)
        & (frame["P_target"] <= frame["P_in"] + 1e-10)
        & (frame["P_in"] <= frame["P_d"] + 1e-10)
        & (frame["P_target"] <= frame["P_d"] + 1e-10)
        & (frame["Tmin_d"] <= frame["Tmax_d"])
    )


def split_rows(
    frame: pd.DataFrame,
    split: str,
    seed: int,
    station_column: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(frame))
    if split == "random":
        train_val, test = train_test_split(
            indices, test_size=0.30, random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=0.20, random_state=seed
        )
        return train, val, test

    if station_column is None:
        raise ValueError("station split requires a station identifier column")
    if frame[station_column].nunique() < 4:
        raise ValueError("station split requires at least four distinct stations")

    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_val_pos, test_pos = next(
        outer.split(indices, groups=frame[station_column].to_numpy())
    )
    train_val = indices[train_val_pos]
    test = indices[test_pos]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    inner_train_pos, val_pos = next(
        inner.split(
            train_val,
            groups=frame.iloc[train_val][station_column].to_numpy(),
        )
    )
    return train_val[inner_train_pos], train_val[val_pos], test


def make_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def predict(model: nn.Module, x: np.ndarray, batch_size: int = 65_536) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            tensor = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    correlation = float("nan")
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "n": int(len(observed)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r": correlation,
        "r2": float(r2_score(observed, predicted)),
        "negative_prediction_fraction": float(np.mean(predicted < 0)),
    }


def threshold_metrics(
    observed: np.ndarray, predicted: np.ndarray
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for threshold in (0.0, 0.1, 0.2, 0.3):
        mask = observed > threshold if threshold == 0 else observed >= threshold
        if mask.sum() >= 2:
            key = f"target_{'gt' if threshold == 0 else 'ge'}_{threshold:g}"
            results[key] = regression_metrics(observed[mask], predicted[mask])
    return results


def target_summary(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(values.count()),
        "zero_fraction": float((values == 0).mean()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
        "minimum": float(values.min()),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "maximum": float(values.max()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    frame = pd.read_csv(args.data)
    features, target, station_column, input_column, schema = detect_schema(
        frame, args.task, args.spatial
    )
    required = [*features, target]
    if args.split == "station":
        if station_column is None:
            raise ValueError("station split requires a station identifier column")
        required.append(station_column)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    diagnostics = cleaning_diagnostics(frame, schema)
    missing_mask = frame[required].isna().any(axis=1)
    diagnostics["missing_required_rows"] = int(missing_mask.sum())
    frame = frame.loc[~missing_mask].copy()
    if not args.keep_physical_inconsistencies:
        keep = physical_consistency_mask(frame, schema)
        diagnostics["removed_union_rows"] = int((~keep).sum())
        frame = frame.loc[keep].copy()
    frame = frame.reset_index(drop=True)
    diagnostics["final_rows"] = int(len(frame))
    if frame.empty:
        raise ValueError("No observations remain after quality control")

    train_idx, val_idx, test_idx = split_rows(
        frame, args.split, args.seed, station_column
    )
    x = frame[features].to_numpy(dtype=np.float64)
    y = frame[[target]].to_numpy(dtype=np.float64)

    x_scaler = StandardScaler().fit(x[train_idx])
    y_scaler = StandardScaler().fit(y[train_idx])
    x_scaled = x_scaler.transform(x)
    y_scaled = y_scaler.transform(y)

    train_loader = make_loader(
        x_scaled[train_idx], y_scaled[train_idx], args.batch_size, True
    )
    val_x = torch.as_tensor(x_scaled[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y_scaled[val_idx], dtype=torch.float32)

    model = PrecipitationDownscalingANN(n_features=len(features), hidden_size=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = nn.MSELoss()

    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(batch_x)
            total_items += len(batch_x)

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(val_x), val_y).item())
        history.append(
            {
                "epoch": epoch,
                "training_loss": total_loss / total_items,
                "validation_loss": val_loss,
            }
        )

        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    model.load_state_dict(best_state)
    predicted_scaled = predict(model, x_scaled[test_idx])
    predicted = y_scaler.inverse_transform(predicted_scaled).reshape(-1)
    observed = y[test_idx].reshape(-1)
    test_input = frame.iloc[test_idx][input_column].to_numpy(dtype=float)

    train_input = frame.iloc[train_idx][input_column].to_numpy(dtype=float)
    train_target = y[train_idx].reshape(-1)
    denominator = float(np.dot(train_input, train_input))
    scale = float(np.dot(train_input, train_target) / denominator) if denominator else 0.0
    scale = float(np.clip(scale, 0.0, 1.0))
    scaled_baseline = scale * test_input

    metrics = {
        "ann": regression_metrics(observed, predicted),
        "ann_by_threshold": threshold_metrics(observed, predicted),
        "persistence_baseline": regression_metrics(observed, test_input),
        "duration_scaling_baseline": regression_metrics(observed, scaled_baseline),
        "duration_scaling_alpha": scale,
        "best_epoch": best_epoch,
        "best_validation_loss_standardized": best_loss,
    }

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "task": args.task,
        "features": features,
        "target": target,
        "hidden_size": 10,
        "state_dict": model.state_dict(),
        "x_mean": x_scaler.mean_,
        "x_scale": x_scaler.scale_,
        "y_mean": y_scaler.mean_,
        "y_scale": y_scaler.scale_,
        "split": args.split,
        "seed": args.seed,
    }
    torch.save(checkpoint, args.output / "model.pt")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    summary = {
        "task": args.task,
        "schema": schema,
        "features": features,
        "target": target,
        "cleaning": diagnostics,
        "training_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "testing_samples": int(len(test_idx)),
        "station_count": int(frame[station_column].nunique())
        if station_column is not None
        else None,
        "timestamp_start": str(pd.to_datetime(frame["timestamp"]).min())
        if "timestamp" in frame.columns
        else None,
        "timestamp_end": str(pd.to_datetime(frame["timestamp"]).max())
        if "timestamp" in frame.columns
        else None,
        "target_statistics": target_summary(frame[target]),
    }
    (args.output / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    id_columns = [
        column
        for column in (station_column, "timestamp")
        if column is not None and column in frame.columns
    ]
    prediction_frame = frame.iloc[test_idx][id_columns].copy()
    prediction_frame["input_precipitation"] = test_input
    prediction_frame["observed"] = observed
    prediction_frame["predicted"] = predicted
    prediction_frame["duration_scaling_prediction"] = scaled_baseline
    prediction_frame.to_csv(args.output / "predictions.csv", index=False)

    print(json.dumps({"output": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
