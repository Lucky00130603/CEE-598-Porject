import netCDF4 as nc
import numpy as np
import pandas as pd
import pytest
import torch

from src.climate import climate_run, required_positions, unit_factor
from src.model import PrecipitationDownscalingANN
from src.project import plot_idf
from src.spatial import enrich
from src.train import PAIR_FEATURES, TASKS


def test_three_hour_units_and_timestamp_gaps():
    assert unit_factor("kg m-2 s-1", "pr")[0] == pytest.approx(10800 / 25.4)
    dates = pd.date_range("2001-01-01", periods=3, freq="3h")
    with pytest.raises(ValueError, match="missing 1 timestamps"):
        required_positions(dates.delete(1), dates, "pr")


def test_complete_leap_year_with_windows(tmp_path):
    year = 2000
    climate = tmp_path / "inputs"
    climate.mkdir()
    for variable, unit, count, spacing, value in (
        ("pr", "mm", 366 * 8, 3, 5.08),
        ("tasmax", "K", 366, 24, 295),
        ("tasmin", "K", 366, 24, 275),
    ):
        path = climate / f"{variable}_ECP2_hadcm3_1968_2000_checked.nc"
        with nc.Dataset(path, "w") as ds:
            for axis, size in (("time", count), ("y", 2), ("x", 2)):
                ds.createDimension(axis, size)
            time = ds.createVariable("time", "i4", ("time",))
            time.units = f"hours since {year}-01-01 00:00:00"
            time[:] = np.arange(count) * spacing
            ds.createVariable("lat", "f8", ("y",))[:] = [40, 41]
            ds.createVariable("lon", "f8", ("x",))[:] = [-89, -88]
            field = ds.createVariable(variable, "f4", ("time", "y", "x"))
            field.units = unit
            field[:] = value
            field[2, 0, 0] = np.ma.masked
    weights = tmp_path / "weights"
    for task in TASKS:
        directory = weights / task
        directory.mkdir(parents=True)
        model = PrecipitationDownscalingANN(4, 10)
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        torch.save({
            "task": task, "features": PAIR_FEATURES, "hidden_size": 10,
            "state_dict": model.state_dict(), "x_mean": np.zeros(4), "x_scale": np.ones(4),
            "y_mean": np.array([0.9]), "y_scale": np.array([1]),
        }, directory / "model.pt")
    out = tmp_path / "results"
    path = climate_run(climate, "ecp2", "historical", (year, year), "start",
                       weights, out, lambda message: None, True)
    with nc.Dataset(path) as ds:
        assert list(ds["year"][:]) == [year]
        np.testing.assert_allclose(ds["annual_max_depth"][0, :, 1, 1], np.full(5, 0.2))
        assert np.ma.is_masked(ds["annual_max_depth"][0, 0, 0, 0]) or np.isnan(ds["annual_max_depth"][0, 0, 0, 0])
    with nc.Dataset(out / "window_depths.nc") as ds:
        assert len(ds["time"]) == 366 * 8
    plot_idf([path], out, 41, -88)
    assert (out / "empirical_idf.png").stat().st_size > 0
    curve = pd.read_csv(out / "empirical_idf.csv")
    assert curve.valid_years.tolist() == [1] * 5
    assert curve.intensity_in_per_hour.iloc[-1] == pytest.approx(0.8)


def test_spatial_predictors_require_simultaneous_neighbors():
    metadata = pd.DataFrame({
        "STATION": list("ABCDEF"), "latitude": np.arange(6) * 0.1 + 40,
        "longitude": [-88] * 6, "elevation": [200] * 6,
    })
    pairs = pd.DataFrame({
        "STATION": list("ABCDEF") + ["A"], "timestamp": ["day1"] * 6 + ["day2"],
        "P_in": np.arange(7, dtype=float),
    })
    result = enrich(pairs, metadata)
    assert result.timestamp.tolist() == ["day1"] * 6
    assert result.loc[result.STATION == "A", "neighbor_max"].item() == 5
