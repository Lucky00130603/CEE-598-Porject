# CEE 598 Project: Temporal Downscaling of Precipitation in Illinois

This repository contains the data-processing and neural-network framework for a CEE 598 project on temporal downscaling of precipitation in Illinois.

## Project objective

The objective is to train artificial neural networks (ANNs) with NOAA observations and then apply the trained models to 3-hour NARCCAP climate-model precipitation. Four regression models are trained independently and applied sequentially:

```text
3-hour precipitation
        -> ANN 1 -> 2-hour precipitation
        -> ANN 2 -> 1-hour precipitation
        -> ANN 3 -> 30-minute precipitation
        -> ANN 4 -> 15-minute precipitation
```

For each model, the baseline predictors are:

1. precipitation at the preceding longer duration;
2. daily precipitation;
3. daily maximum temperature; and
4. daily minimum temperature.

The target is precipitation depth at the finer duration. This is a regression project, not a classification project.

## Current status

- The PyTorch model architecture and reproducible training pipeline are implemented.
- The four temporal-downscaling tasks are configured.
- Metrics, early stopping, prediction export, and train/validation/test splitting are implemented.
- NOAA preprocessing and final input-target table construction are in progress.
- No final scientific results are claimed in this repository yet.

## ANN architecture

The baseline neural network has the following architecture:

```text
4 input features -> 10 sigmoid hidden neurons -> 1 linear output
```

The model is implemented in PyTorch and trained using mean squared error, Adam optimization, and validation-based early stopping.

## Spatial extension

A feedforward ANN processes each station-time observation independently. An optional spatial-feature model adds:

- station or grid-cell latitude, longitude, and elevation;
- inverse-distance-weighted precipitation from nearby locations;
- neighboring precipitation maximum; and
- neighboring precipitation standard deviation.

The spatial model will be evaluated against the four-input baseline. Station-based holdout splitting is supported so that the spatial generalization test does not place the same station in both training and testing data.

## Expected processed data

The training script expects one CSV containing aligned NOAA observations. Required columns are:

```text
station_id,timestamp,p_3h,p_2h,p_1h,p_30min,p_15min,
daily_precip,tmax,tmin
```

Optional spatial columns are:

```text
latitude,longitude,elevation,
neighbor_idw_mean,neighbor_max,neighbor_std
```

Column names can be changed in the source code if the final NOAA export uses a different convention. Raw NOAA and NARCCAP files are intentionally excluded from Git because they are large and may have distribution constraints. See [`data/README.md`](data/README.md).

## Repository structure

```text
.
├── data/README.md          # Data sources, schema, and preprocessing contract
├── scripts/make_demo_data.py # Generate a small synthetic test dataset
├── scripts/train_all.py    # Train all four models sequentially
├── src/model.py            # Feedforward precipitation ANN
├── src/train.py            # Splitting, standardization, training, and evaluation
├── tests/test_model.py     # Basic model tests
├── requirements.txt
└── .github/workflows/ci.yml
```

## Setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Train one downscaling model

Baseline model:

```bash
python -m src.train \
  --data data/processed/training_table.csv \
  --task 3h_to_2h \
  --output results/3h_to_2h
```

Spatial-feature model:

```bash
python -m src.train \
  --data data/processed/training_table.csv \
  --task 3h_to_2h \
  --spatial \
  --split station \
  --output results/3h_to_2h_spatial
```

Available task names are `3h_to_2h`, `2h_to_1h`, `1h_to_30min`, and `30min_to_15min`.

## Train all four models

```bash
python scripts/train_all.py \
  --data data/processed/training_table.csv \
  --output-root results/baseline
```

Add `--spatial` to train the spatial-feature versions.

## Test the pipeline with synthetic data

The following commands generate a small artificial dataset and verify that the training pipeline runs from beginning to end. The generated values are for software testing only and are not scientific results.

```bash
python scripts/make_demo_data.py \
  --output data/processed/demo_training_table.csv

python -m src.train \
  --data data/processed/demo_training_table.csv \
  --task 3h_to_2h \
  --epochs 10 \
  --output results/demo
```

Each run writes:

- `model.pt`: trained parameters and standardization statistics;
- `metrics.json`: MAE, RMSE, correlation, R-squared, and negative-prediction rate;
- `predictions.csv`: observed and predicted test values;
- `history.csv`: training and validation losses; and
- `data_summary.json`: final sample counts and target statistics.

## Planned evaluation

Performance will be evaluated with MAE, RMSE, Pearson correlation, and R-squared. Errors will also be reported for larger precipitation observations because annual maxima and IDF analysis are sensitive to the upper tail. Predictions will be checked for negative or otherwise physically unreasonable values.

After evaluation on held-out NOAA observations, the four trained models will be applied sequentially to Illinois NARCCAP historical (1971-2000) and future (2041-2070) simulations.
