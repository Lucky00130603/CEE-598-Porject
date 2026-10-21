import pytest
import torch

from src.model import PrecipitationDownscalingANN


def test_baseline_output_shape() -> None:
    model = PrecipitationDownscalingANN(n_features=4, hidden_size=10)
    output = model(torch.zeros(8, 4))
    assert output.shape == (8, 1)


def test_spatial_output_shape() -> None:
    model = PrecipitationDownscalingANN(n_features=10, hidden_size=10)
    output = model(torch.zeros(5, 10))
    assert output.shape == (5, 1)


def test_invalid_feature_count() -> None:
    with pytest.raises(ValueError):
        PrecipitationDownscalingANN(n_features=0)
