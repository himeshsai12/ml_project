"""Training, evaluation, prediction, and model persistence."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import RANDOM_STATE

MODEL_FACTORIES = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=180, max_depth=18, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(
        n_estimators=160, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE
    ),
}


@dataclass(frozen=True)
class Evaluation:
    model_name: str
    mae: float
    rmse: float
    r2: float
    trained_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_model(model_name: str) -> Pipeline:
    """Create a leakage-resistant model pipeline."""
    if model_name not in MODEL_FACTORIES:
        raise ValueError(f"Unknown model: {model_name}")
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("regressor", MODEL_FACTORIES[model_name]()),
        ]
    )


def train_and_evaluate(
    model_name: str,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Pipeline, Evaluation, pd.DataFrame]:
    """Fit one model and return metrics plus row-level predictions."""
    model = build_model(model_name)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    errors = pd.DataFrame(
        {
            "actual": y_test.to_numpy(),
            "predicted": predictions,
            "residual": y_test.to_numpy() - predictions,
        },
        index=y_test.index,
    ).sort_values("residual")
    evaluation = Evaluation(
        model_name=model_name,
        mae=float(mean_absolute_error(y_test, predictions)),
        rmse=float(np.sqrt(mean_squared_error(y_test, predictions))),
        r2=float(r2_score(y_test, predictions)),
        trained_at=datetime.now(timezone.utc).isoformat(),
    )
    return model, evaluation, errors


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f"No trained model found at {path}")
    return joblib.load(path)
