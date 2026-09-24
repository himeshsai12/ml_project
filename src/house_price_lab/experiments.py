"""Persistence for model comparison runs."""

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .modeling import Evaluation

DEFAULT_EXPERIMENT_DIR = Path("experiments")


def save_experiment(
    evaluation: Evaluation,
    directory: Path = DEFAULT_EXPERIMENT_DIR,
    settings: dict[str, Any] | None = None,
) -> Path:
    """Save a run as a readable JSON record and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    record = {
        "id": timestamp.strftime("%Y%m%dT%H%M%S%fZ"),
        "saved_at": timestamp.isoformat(),
        "evaluation": asdict(evaluation),
        "settings": settings or {},
    }
    path = directory / f"{record['id']}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def load_experiments(directory: Path = DEFAULT_EXPERIMENT_DIR) -> list[dict[str, Any]]:
    """Load saved runs newest first."""
    if not directory.exists():
        return []
    records = []
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return records
