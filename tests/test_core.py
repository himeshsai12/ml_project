from pathlib import Path

import pandas as pd

from src.house_price_lab.data import load_dataset, split_dataset
from src.house_price_lab.experiments import load_experiments, save_experiment
from src.house_price_lab.modeling import Evaluation, train_and_evaluate


def test_dataset_has_expected_shape_and_target():
    dataset = load_dataset()
    assert dataset.features.shape == (20640, 8)
    assert dataset.target.name == "median_house_value"
    assert dataset.target_unit == "$100,000"


def test_split_is_reproducible():
    dataset = load_dataset()
    first = split_dataset(dataset)
    second = split_dataset(dataset)
    for first_part, second_part in zip(first, second):
        pd.testing.assert_frame_equal(first_part, second_part) if isinstance(first_part, pd.DataFrame) else pd.testing.assert_series_equal(first_part, second_part)


def test_model_training_returns_metrics_and_residuals():
    dataset = load_dataset()
    x_train, x_test, y_train, y_test = split_dataset(dataset)
    _, evaluation, errors = train_and_evaluate(
        "Linear Regression", x_train, x_test, y_train, y_test
    )
    assert evaluation.model_name == "Linear Regression"
    assert 0 < evaluation.rmse < 2
    assert -1 < evaluation.r2 < 1
    assert list(errors.columns) == ["actual", "predicted", "residual"]
    assert len(errors) == len(y_test)


def test_experiment_round_trip(tmp_path: Path):
    evaluation = Evaluation(
        model_name="Test Model",
        mae=0.1,
        rmse=0.2,
        r2=0.8,
        trained_at="2026-01-01T00:00:00+00:00",
    )
    saved_path = save_experiment(evaluation, tmp_path, {"seed": 42})
    records = load_experiments(tmp_path)
    assert saved_path.exists()
    assert records[0]["evaluation"]["model_name"] == "Test Model"
    assert records[0]["settings"]["seed"] == 42
