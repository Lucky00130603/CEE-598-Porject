# CEE 598 Project: Temporal Downscaling of Precipitation in Illinois

This repository contains the data-processing, training, and evaluation workflow for a CEE 598 project on temporal downscaling of precipitation in Illinois.

## Project objective

Four regression models are trained independently and will ultimately be applied sequentially:

```text
3-hour precipitation
        -> ANN 1 -> 2-hour precipitation
        -> ANN 2 -> 1-hour precipitation
        -> ANN 3 -> 30-minute precipitation
        -> ANN 4 -> 15-minute precipitation
```

Each model uses four predictors: precipitation at the preceding longer duration, daily precipitation, daily maximum temperature, and daily minimum temperature. Temperature is retained because it provides daily meteorological context and supports the model specification adopted by the project team.

## Current progress

- Four NOAA observation-pair tables have been assembled and quality controlled.
- The tables contain 1,630,093 raw records and 1,594,662 retained records from 49-52 Illinois stations.
- All four baseline ANNs have completed preliminary training.
- The held-out test-set R-squared values range from 0.902 to 0.965.
- Reproducible data summaries, metrics, diagnostics, and figures are available in [`results/report2`](results/report2).
- Spatial predictors and application to NARCCAP simulations remain future work; the current paired files do not yet include station coordinates or neighboring-station features.

These are preliminary progress-report results, not finalized scientific estimates.

## Data schema and quality control

The training code directly supports one CSV for each downscaling task. Required columns are:

```text
STATION,timestamp,P_in,P_target,P_d,Tmax_d,Tmin_d
```

`P_in` is the preceding longer-duration precipitation, `P_target` is the finer-duration target, and `P_d`, `Tmax_d`, and `Tmin_d` are the daily precipitation, maximum temperature, and minimum temperature variables.

By default, training removes rows with missing required values, negative precipitation, target precipitation greater than input precipitation, subdaily precipitation greater than daily precipitation, or daily minimum temperature greater than daily maximum temperature. The number of rows affected by each check is recorded in `data_summary.json`.

## ANN and evaluation

The baseline network is:

```text
4 input features -> 10 sigmoid hidden neurons -> 1 linear output
```

The implementation uses PyTorch, mean squared error, Adam optimization, train-only standardization, validation-based model selection, and a reproducible random 56/14/30 percent train/validation/test split. Each run reports MAE, RMSE, Pearson correlation, R-squared, large-event metrics, a persistence baseline, and a fitted duration-scaling baseline.

## Repository structure

```text
.
├── data/                   # Data notes and the uploaded NOAA archive
├── results/report2/        # Lightweight summaries and report figures
├── scripts/train_all.py    # Train the four paired-data models
├── scripts/summarize_report2.py
├── src/model.py            # Feedforward precipitation ANN
├── src/train.py            # Quality control, training, and evaluation
├── tests/                  # Unit tests
└── .github/workflows/ci.yml
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train one model

```bash
python -m src.train \
  --data data/processed/pairs_3h_to_2h.csv \
  --task 3h_to_2h \
  --epochs 60 \
  --patience 8 \
  --batch-size 4096 \
  --learning-rate 0.003 \
  --output results/training/3h_to_2h
```

Available task names are `3h_to_2h`, `2h_to_1h`, `1h_to_30min`, and `30min_to_15min`.

## Train all four models

Place the four `pairs_*.csv` files in one directory, then run:

```bash
python scripts/train_all.py \
  --data-dir data/processed \
  --epochs 60 \
  --patience 8 \
  --batch-size 4096 \
  --learning-rate 0.003 \
  --output-root results/training
```

Each task writes `model.pt`, `metrics.json`, `predictions.csv`, `history.csv`, and `data_summary.json`. Model weights and row-level predictions are intentionally excluded from version control; the compact summaries in `results/report2` are tracked.

## Reproduce the Report II summaries

```bash
python scripts/summarize_report2.py \
  --output-root results/training \
  --summary-dir results/report2
```

## Run tests

```bash
pytest -q
```
