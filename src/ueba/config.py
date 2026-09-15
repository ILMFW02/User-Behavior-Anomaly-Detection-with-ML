"""Configuration loading and project path handling."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    answers_dir: Path
    processed_dir: Path
    model_dir: Path
    report_dir: Path


@dataclass(frozen=True)
class FeatureSettings:
    workday_start_hour: int
    workday_end_hour: int
    suspicious_url_terms: tuple[str, ...]
    job_search_url_terms: tuple[str, ...]
    data_leak_url_terms: tuple[str, ...]
    monitoring_tool_url_terms: tuple[str, ...]
    sensitive_file_terms: tuple[str, ...]


@dataclass(frozen=True)
class BaselineSettings:
    min_history_days: int
    rolling_window_days: int


@dataclass(frozen=True)
class ModelSettings:
    peer_clusters: int
    isolation_forest_estimators: int
    contamination: float
    anomaly_weight: float
    peer_weight: float
    random_state: int
    model_family: str = "isolation_forest"
    lof_neighbors: int = 35
    ocsvm_nu: float = 0.01
    ocsvm_gamma: str = "scale"


@dataclass(frozen=True)
class SelectionSettings:
    correlation_threshold: float = 0.90
    minimum_features: int = 20
    maximum_features: int = 30
    permutation_repetitions: int = 5


@dataclass(frozen=True)
class EvaluationSettings:
    alert_k: int = 10
    top_q: int = 3
    early_warning_days: int = 7


@dataclass(frozen=True)
class TemporalSettings:
    initial_train_days: int = 180
    validation_days: int = 60
    test_days: int = 90
    step_days: int = 90
    embargo_days: int = 0


@dataclass(frozen=True)
class DriftSettings:
    method: str = "page_hinkley"
    delta: float = 0.005
    threshold: float = 20.0
    minimum_instances: int = 30


@dataclass(frozen=True)
class Settings:
    paths: Paths
    features: FeatureSettings
    baseline: BaselineSettings
    model: ModelSettings
    selection: SelectionSettings
    evaluation: EvaluationSettings
    temporal: TemporalSettings
    drift: DriftSettings


def load_settings(config_path: str | Path) -> Settings:
    """Load a TOML config, resolving relative paths from its parent directory."""
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as config_file:
        data = tomllib.load(config_file)

    root = config_path.parent
    path_data = data["paths"]

    def resolve(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    return Settings(
        paths=Paths(
            raw_dir=resolve(path_data["raw_dir"]),
            answers_dir=resolve(path_data["answers_dir"]),
            processed_dir=resolve(path_data["processed_dir"]),
            model_dir=resolve(path_data["model_dir"]),
            report_dir=resolve(path_data["report_dir"]),
        ),
        features=FeatureSettings(
            workday_start_hour=int(data["features"]["workday_start_hour"]),
            workday_end_hour=int(data["features"]["workday_end_hour"]),
            suspicious_url_terms=tuple(data["features"]["suspicious_url_terms"]),
            job_search_url_terms=tuple(data["features"]["job_search_url_terms"]),
            data_leak_url_terms=tuple(data["features"]["data_leak_url_terms"]),
            monitoring_tool_url_terms=tuple(data["features"]["monitoring_tool_url_terms"]),
            sensitive_file_terms=tuple(data["features"]["sensitive_file_terms"]),
        ),
        baseline=BaselineSettings(
            min_history_days=int(data["baseline"]["min_history_days"]),
            rolling_window_days=int(data["baseline"]["rolling_window_days"]),
        ),
        model=ModelSettings(
            peer_clusters=int(data["model"]["peer_clusters"]),
            isolation_forest_estimators=int(data["model"]["isolation_forest_estimators"]),
            contamination=float(data["model"]["contamination"]),
            anomaly_weight=float(data["model"]["anomaly_weight"]),
            peer_weight=float(data["model"]["peer_weight"]),
            random_state=int(data["model"]["random_state"]),
            model_family=str(data["model"].get("model_family", "isolation_forest")),
            lof_neighbors=int(data["model"].get("lof_neighbors", 35)),
            ocsvm_nu=float(data["model"].get("ocsvm_nu", data["model"]["contamination"])),
            ocsvm_gamma=str(data["model"].get("ocsvm_gamma", "scale")),
        ),
        selection=SelectionSettings(
            correlation_threshold=float(data.get("selection", {}).get("correlation_threshold", 0.90)),
            minimum_features=int(data.get("selection", {}).get("minimum_features", 20)),
            maximum_features=int(data.get("selection", {}).get("maximum_features", 30)),
            permutation_repetitions=int(data.get("selection", {}).get("permutation_repetitions", 5)),
        ),
        evaluation=EvaluationSettings(
            alert_k=int(data.get("evaluation", {}).get("alert_k", 10)),
            top_q=int(data.get("evaluation", {}).get("top_q", 3)),
            early_warning_days=int(data.get("evaluation", {}).get("early_warning_days", 7)),
        ),
        temporal=TemporalSettings(
            initial_train_days=int(data.get("temporal", {}).get("initial_train_days", 180)),
            validation_days=int(data.get("temporal", {}).get("validation_days", 60)),
            test_days=int(data.get("temporal", {}).get("test_days", 90)),
            step_days=int(data.get("temporal", {}).get("step_days", 90)),
            embargo_days=int(data.get("temporal", {}).get("embargo_days", 0)),
        ),
        drift=DriftSettings(
            method=str(data.get("drift", {}).get("method", "page_hinkley")),
            delta=float(data.get("drift", {}).get("delta", 0.005)),
            threshold=float(data.get("drift", {}).get("threshold", 20.0)),
            minimum_instances=int(data.get("drift", {}).get("minimum_instances", 30)),
        ),
    )
