"""Dataset loading and deterministic data preparation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TARGET_NAME = "median_house_value"
TARGET_UNIT = "$100,000"


@dataclass(frozen=True)
class DatasetBundle:
    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]
    target_name: str = TARGET_NAME
    target_unit: str = TARGET_UNIT


def load_dataset() -> DatasetBundle:
    """Load the bundled California Housing regression dataset."""
    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame.copy()
    target = frame.pop("MedHouseVal").rename(TARGET_NAME)
    features = frame.rename(columns={"MedInc": "median_income"})
    return DatasetBundle(
        features=features,
        target=target,
        feature_names=tuple(features.columns),
    )


def split_dataset(
    bundle: DatasetBundle, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible train/test split."""
    return train_test_split(
        bundle.features,
        bundle.target,
        test_size=test_size,
        random_state=RANDOM_STATE,
    )
