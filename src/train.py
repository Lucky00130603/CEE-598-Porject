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
        default="station",
        help="Use station holdout by default to test spatial generalization.",
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


def split_rows(
    frame: pd.DataFrame, split: str, seed: int
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

    if "station_id" not in frame.columns:
        raise ValueError("station split requires a station_id column")
    if frame["station_id"].nunique() < 4:
        raise ValueError("station split requires at least four distinct stations")

    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_val_pos, test_pos = next(
        outer.split(indices, groups=frame["station_id"].to_numpy())
    )
    train_val = indices[train_val_pos]
    test = indices[test_pos]

    inner = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    inner_train_pos, val_pos = next(
        inner.split(
            train_val,
            groups=frame.iloc[train_val]["station_id"].to_numpy(),
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


def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(x, dtype=torch.float32)
        return model(tensor).cpu().numpy()


def regression_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed).reshape(-1)
    predicted = np.asarray(predicted).reshape(-1)
    correlation = float("nan")
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(mean_squared_error(observed, predicted) ** 0.5),
        "r": correlation,
        "r2": float(r2_score(observed, predicted)),
        "negative_prediction_fraction": float(np.mean(predicted < 0)),
    }


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

    source_feature, target = TASKS[args.task]
    features = [source_feature, *DAILY_FEATURES]
    if args.spatial:
        features.extend(SPATIAL_FEATURES)

    frame = pd.read_csv(args.data)
    required = [*features, target]
    if args.split == "station":
        required.append("station_id")
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame.dropna(subset=required).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No complete observations remain after dropping missing values")

    train_idx, val_idx, test_idx = split_rows(frame, args.split, args.seed)
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
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(val_x), val_y).item())
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(batch_losses)),
                "validation_loss": val_loss,
            }
        )

        if val_loss < best_loss - 1e-8:
            best_loss = val_loss
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
    metrics = regression_metrics(observed, predicted)

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
        "features": features,
        "target": target,
        "total_complete_samples": int(len(frame)),
        "training_samples": int(len(train_idx)),
        "validation_samples": int(len(val_idx)),
        "testing_samples": int(len(test_idx)),
        "station_count": int(frame["station_id"].nunique())
        if "station_id" in frame.columns
        else None,
        "target_statistics": target_summary(frame[target]),
    }
    (args.output / "data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
    prediction_frame = frame.iloc[test_idx][
        [column for column in ("station_id", "timestamp") if column in frame.columns]
    ].copy()
    prediction_frame["observed"] = observed
    prediction_frame["predicted"] = predicted
    prediction_frame.to_csv(args.output / "predictions.csv", index=False)

    print(json.dumps({"output": str(args.output), **metrics}, indent=2))


if __name__ == "__main__":
    main()
