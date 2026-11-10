import pandas as pd

from src.train import PAIR_FEATURES, detect_schema, physical_consistency_mask


def paired_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STATION": [1, 1, 2, 2, 3],
            "timestamp": pd.date_range("2000-01-01", periods=5, freq="h"),
            "P_in": [1.0, 0.5, 0.2, 0.4, 0.3],
            "P_target": [0.8, 0.6, 0.1, 0.2, 0.1],
            "P_d": [1.2, 1.0, 0.1, 0.8, 0.8],
            "Tmax_d": [70.0, 70.0, 65.0, 60.0, 55.0],
            "Tmin_d": [50.0, 50.0, 45.0, 65.0, 40.0],
        }
    )


def test_detect_paired_schema_retains_temperature() -> None:
    features, target, station, input_column, schema = detect_schema(
        paired_frame(), "3h_to_2h", spatial=False
    )
    assert features == PAIR_FEATURES
    assert features[-2:] == ["Tmax_d", "Tmin_d"]
    assert (target, station, input_column, schema) == (
        "P_target",
        "STATION",
        "P_in",
        "paired",
    )


def test_physical_quality_control_flags_expected_rows() -> None:
    mask = physical_consistency_mask(paired_frame(), "paired")
    assert mask.tolist() == [True, False, False, False, True]
