"""Audit NetCDF inputs and estimate nested duration depths per NARCCAP window."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path

import netCDF4 as nc
import numpy as np
import pandas as pd
import torch

from .model import PrecipitationDownscalingANN
from .train import PAIR_FEATURES, TASKS


MODELS = {
    "ecp2": "ECP2_hadcm3",
    "hrm3": "HRM3_gfdl",
    "crcm": "CRCM_cgcm3",
    "mm5i": "MM5I_ccsm",
}
PERIODS = {"historical": (1971, 2000), "future": (2041, 2070)}
DURATIONS = (180, 120, 60, 30, 15)


def climate_files(folder, model, period):
    start = "1968" if period == "historical" else "2038"
    files = {}
    for variable in ("pr", "tasmax", "tasmin"):
        pattern = f"{variable}_{MODELS[model]}_{start}_*_checked.nc"
        candidates = sorted(folder.rglob(pattern))
        if len(candidates) != 1:
            raise ValueError(f"Expected one {pattern} in {folder}; found {len(candidates)}")
        files[variable] = candidates[0]
    return files


def read_values(variable, key=slice(None)):
    return np.ma.asarray(variable[key], dtype=np.float64).filled(np.nan)


def read_dates(variable, key=slice(None), units=None, calendar=None):
    calendar = calendar or getattr(variable, "calendar", "standard")
    if calendar not in {"standard", "gregorian", "proleptic_gregorian"}:
        raise ValueError(f"Calendar {calendar} requires an explicitly corrected file")
    values = nc.num2date(
        read_values(variable, key), units or variable.units, calendar=calendar,
        only_use_cftime_datetimes=False, only_use_python_datetimes=True,
    )
    return pd.DatetimeIndex(values)


def coordinates(dataset, dims):
    if {"lat_wgs84", "lon_wgs84"} <= dataset.variables.keys():
        names = ("lat_wgs84", "lon_wgs84")
    else:
        names = ("lat", "lon")
    lat, lon = (dataset.variables.get(name) for name in names)
    if lat is None or lon is None:
        raise ValueError("Expected geographic latitude/longitude coordinates")
    if lat.dimensions == dims and lon.dimensions == dims:
        return read_values(lat), read_values(lon)
    if lat.dimensions == dims[:1] and lon.dimensions == dims[1:]:
        return tuple(np.meshgrid(read_values(lat), read_values(lon), indexing="ij"))
    raise ValueError(f"Coordinate dimensions {lat.dimensions}, {lon.dimensions} do not match {dims}")


def required_positions(actual, expected, label):
    if actual.has_duplicates or not actual.is_monotonic_increasing:
        raise ValueError(f"{label}: duplicate or unordered timestamps")
    positions = actual.get_indexer(expected)
    if (positions < 0).any():
        missing = expected[positions < 0]
        raise ValueError(f"{label}: missing {len(missing)} timestamps; first is {missing[0]}")
    return positions


def read_axes(datasets, stamp):
    axes = {}
    reference = None
    for name, ds in datasets.items():
        variable = ds.variables[name]
        if len(variable.dimensions) != 3:
            raise ValueError(f"{name}: expected time and two grid axes, got {variable.dimensions}")
        time = ds.variables[variable.dimensions[0]]
        grid = coordinates(ds, variable.dimensions[1:])
        if reference is None:
            reference = grid
        elif not all(a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-4, equal_nan=True)
                     for a, b in zip(grid, reference)):
            raise ValueError(f"{name}: grid is not aligned with precipitation")
        dates = read_dates(time)
        bounds_name = getattr(time, "bounds", "")
        if name == "pr" and bounds_name in ds.variables:
            bounds = ds.variables[bounds_name]
            calendar = getattr(time, "calendar", "standard")
            starts = read_dates(bounds, (slice(None), 0), time.units, calendar)
            ends = read_dates(bounds, (slice(None), 1), time.units, calendar)
            if not ((ends - starts) == pd.Timedelta(hours=3)).all():
                raise ValueError("Precipitation time bounds are not exactly three hours")
            dates = starts
        elif name == "pr":
            offsets = {"start": 0, "center": 90, "end": 180}
            if stamp is None:
                raise ValueError("pr has no time bounds; verify source metadata and pass --stamp start, center, or end")
            dates -= pd.Timedelta(minutes=offsets[stamp])
        else:
            dates = dates.normalize()
        axes[name] = dates
    return axes, reference


def unit_factor(units, quantity):
    unit = units.lower().replace(" ", "").replace("**", "").replace("^", "")
    if quantity == "pr":
        factors = {
            "kgm-2s-1": 10800 / 25.4, "kg/m2/s": 10800 / 25.4,
            "mms-1": 10800 / 25.4, "mm/s": 10800 / 25.4,
            "kgm-2": 1 / 25.4, "kg/m2": 1 / 25.4, "mm": 1 / 25.4,
            "m": 1000 / 25.4, "in": 1.0, "inch": 1.0,
        }
        if unit in factors:
            return factors[unit], 0.0
    else:
        if unit in {"k", "kelvin"}:
            return 1.8, 32 - 273.15 * 1.8
        if unit in {"c", "degc", "degree_celsius", "degrees_celsius", "celsius"}:
            return 1.8, 32.0
        if unit in {"f", "degf", "degree_fahrenheit", "degrees_fahrenheit", "fahrenheit"}:
            return 1.0, 0.0
    raise ValueError(f"Unsupported units for {quantity}: {units!r}; verify the source")


def load_models(root):
    models = []
    for task in TASKS:
        path = root / task / "model.pt"
        # The trainer saves NumPy scalers. Only load checkpoints produced by this project.
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck["task"] != task or list(ck["features"]) != PAIR_FEATURES:
            raise ValueError(f"{path} does not contain the expected four-input baseline")
        model = PrecipitationDownscalingANN(len(PAIR_FEATURES), ck["hidden_size"])
        model.load_state_dict(ck["state_dict"])
        models.append((model.eval(), ck))
    return models


@torch.inference_mode()
def cascade(coarse, daily, tmax, tmin, models, batch=65536):
    wet = np.flatnonzero(np.isfinite(coarse.ravel()) & (coarse.ravel() > 0))
    previous = coarse
    for model, ck in models:
        prediction = previous.copy().ravel()
        for start in range(0, wet.size, batch):
            selected = wet[start:start + batch]
            features = np.column_stack([v.ravel()[selected] for v in (previous, daily, tmax, tmin)])
            x = torch.as_tensor((features - ck["x_mean"]) / ck["x_scale"], dtype=torch.float32)
            raw = model(x).numpy().ravel() * ck["y_scale"][0] + ck["y_mean"][0]
            prediction[selected] = np.clip(raw, 0, previous.ravel()[selected])
        previous = prediction.reshape(coarse.shape)
        yield previous


def climate_run(folder, model, period, years, stamp, weights, out, status, save_windows=False):
    files = climate_files(folder, model, period)
    out.mkdir(parents=True, exist_ok=True)
    max_final = out / "annual_maxima.nc"
    window_final = out / "window_depths.nc"
    max_temp = out / "annual_maxima.partial.nc"
    window_temp = out / "window_depths.partial.nc"
    chain = load_models(weights)
    with ExitStack() as stack:
        inputs = {key: stack.enter_context(nc.Dataset(path)) for key, path in files.items()}
        axes, (lat, lon) = read_axes(inputs, stamp)
        variables = {key: ds[key] for key, ds in inputs.items()}
        conversions = {key: unit_factor(var.units, key) for key, var in variables.items()}
        # Validate the full requested timestamp range before creating outputs.
        for name, dates in axes.items():
            freq = "3h" if name == "pr" else "D"
            expected = pd.date_range(f"{years[0]}-01-01", f"{years[1]+1}-01-01", freq=freq, inclusive="left")
            required_positions(dates, expected, name)
            status(f"{model}/{period}: {name} {variables[name].units}, {dates[0]} to {dates[-1]}")
        maxes = stack.enter_context(nc.Dataset(max_temp, "w"))
        results = [maxes]
        if save_windows:
            windows = stack.enter_context(nc.Dataset(window_temp, "w"))
            results.append(windows)
        for ds in results:
            ds.description = "Nested duration depths within each 3-hour window"
            ds.climate_model = MODELS[model]
            ds.period = period
            for key, path in files.items():
                ds.setncattr(f"source_{key}", path.name)
            for dimension, size in (("y", lat.shape[0]), ("x", lat.shape[1]), ("duration", len(DURATIONS))):
                ds.createDimension(dimension, size)
            ds.createVariable("duration_minutes", "i4", ("duration",))[:] = DURATIONS
            ds.createVariable("lat", "f8", ("y", "x"))[:] = lat
            ds.createVariable("lon", "f8", ("y", "x"))[:] = lon
        maxes.createDimension("year", years[1] - years[0] + 1)
        maxes.createVariable("year", "i4", ("year",))[:] = np.arange(years[0], years[1] + 1)
        maxima = maxes.createVariable("annual_max_depth", "f4", ("year", "duration", "y", "x"), zlib=True)
        maxima.units = "in"
        coverage = maxes.createVariable("valid_window_fraction", "f4", ("year", "y", "x"), zlib=True)
        if save_windows:
            windows.createDimension("time", None)
            time = windows.createVariable("time", "i8", ("time",))
            time.units = "hours since 1970-01-01 00:00:00"
            depth = windows.createVariable("duration_depth", "f4", ("time", "duration", "y", "x"), zlib=True)
            depth.units = "in"
        offset = 0
        for position, year in enumerate(range(years[0], years[1] + 1)):
            days = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", inclusive="left")
            steps = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="3h", inclusive="left")
            values = {}
            for name, dates in axes.items():
                selected = required_positions(dates, steps if name == "pr" else days, name)
                scale, intercept = conversions[name]
                values[name] = read_values(variables[name], selected) * scale + intercept
            pr = values["pr"]
            pr[(pr < 0) | ~np.isfinite(pr)] = np.nan
            daily = np.repeat(pr.reshape(len(days), 8, *lat.shape).sum(axis=1), 8, axis=0)
            tmax, tmin = (np.repeat(values[name], 8, axis=0) for name in ("tasmax", "tasmin"))
            valid = np.isfinite(daily) & np.isfinite(tmax) & np.isfinite(tmin) & (tmin <= tmax)
            pr = np.where(valid & np.isfinite(lat) & np.isfinite(lon), pr, np.nan)
            year_fraction = np.isfinite(pr).mean(axis=0)
            coverage[position] = year_fraction
            if save_windows:
                time[offset:offset+len(steps)] = (steps - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1)
            maxima[position, 0] = np.where(year_fraction == 1, np.max(pr, axis=0), np.nan)
            if save_windows:
                depth[offset:offset+len(steps), 0] = pr
            for j, estimate in enumerate(cascade(pr, daily, tmax, tmin, chain), 1):
                maxima[position, j] = np.where(year_fraction == 1, np.max(estimate, axis=0), np.nan)
                if save_windows:
                    depth[offset:offset+len(steps), j] = estimate
            offset += len(steps)
            if not np.any(year_fraction == 1):
                raise ValueError(f"{model}/{period} {year}: zero grid cells with a complete valid year")
            status(f"{model}/{period} {year}: {np.count_nonzero(year_fraction == 1)} complete grid cells")
    max_temp.replace(max_final)
    if save_windows:
        window_temp.replace(window_final)
    return max_final


def main():
    parser = argparse.ArgumentParser(description="Inspect NARCCAP filenames, grids, units and timestamp metadata")
    parser.add_argument("--climate-dir", type=Path, default=Path("data/raw/narccap"))
    parser.add_argument("--model", choices=MODELS, default="ecp2")
    parser.add_argument("--period", choices=PERIODS, default="historical")
    parser.add_argument("--stamp", choices=["start", "center", "end"])
    args = parser.parse_args()
    with ExitStack() as stack:
        files = climate_files(args.climate_dir, args.model, args.period)
        datasets = {key: stack.enter_context(nc.Dataset(path)) for key, path in files.items()}
        for name, ds in datasets.items():
            var = ds[name]
            time = ds[var.dimensions[0]]
            dates = read_dates(time)
            print(f"{files[name].name} | shape={var.shape} | units={var.units} | calendar={getattr(time, 'calendar', 'standard')}")
            print(f"  raw first times={dates[:4].tolist()}, last={dates[-1]}")
            print(f"  bounds={getattr(time, 'bounds', 'absent')}, cell_methods={getattr(var, 'cell_methods', 'absent')}")
            unit_factor(var.units, name)
        axes, _ = read_axes(datasets, args.stamp)
        print(f"Aligned grids; first effective precipitation interval starts at {axes['pr'][0]}")


if __name__ == "__main__":
    main()
