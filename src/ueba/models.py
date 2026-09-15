"""Peer grouping and unsupervised anomaly detection."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import OneClassSVM

from .baseline import raw_feature_columns
from .config import ModelSettings


def _matrix(frame: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return frame.select(columns).to_numpy()


class TorchAutoencoder:
    """Small optional reconstruction detector loaded only when PyTorch is installed."""

    def __init__(self, epochs: int = 20, batch_size: int = 256, random_state: int = 42) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.network: object | None = None

    def fit(self, matrix: np.ndarray) -> TorchAutoencoder:
        try:
            import torch
            from torch import nn
        except ImportError as error:
            raise ImportError(
                "Autoencoder requires the optional dependency. Install with `pip install -e '.[autoencoder]'`."
            ) from error
        torch.manual_seed(self.random_state)
        width = matrix.shape[1]
        hidden = max(2, min(32, width // 2 or 2))
        network = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, width))
        data = torch.tensor(matrix, dtype=torch.float32)
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        network.train()
        for _ in range(self.epochs):
            for start in range(0, len(data), self.batch_size):
                batch = data[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = criterion(network(batch), batch)
                loss.backward()
                optimizer.step()
        self.network = network.eval()
        return self

    def score_samples(self, matrix: np.ndarray) -> np.ndarray:
        if self.network is None:
            raise ValueError("Autoencoder must be fit before scoring.")
        import torch

        with torch.no_grad():
            data = torch.tensor(matrix, dtype=torch.float32)
            reconstruction = self.network(data)  # type: ignore[operator]
            return -torch.mean((reconstruction - data) ** 2, dim=1).cpu().numpy()


def make_anomaly_model(settings: ModelSettings, n_samples: int | None = None) -> object:
    """Create a detector whose ``score_samples`` direction is high=normal."""
    family = settings.model_family.lower()
    if family in {"iforest", "isolation_forest"}:
        return IsolationForest(
            n_estimators=settings.isolation_forest_estimators,
            contamination=settings.contamination,
            random_state=settings.random_state,
            n_jobs=-1,
        )
    if family == "lof":
        # LOF requires a neighbour count smaller than the fitted sample count.
        return LocalOutlierFactor(
            n_neighbors=min(settings.lof_neighbors, max(1, (n_samples or settings.lof_neighbors + 1) - 1)),
            contamination=settings.contamination,
            novelty=True,
            n_jobs=-1,
        )
    if family in {"ocsvm", "one_class_svm"}:
        return OneClassSVM(nu=settings.ocsvm_nu, gamma=settings.ocsvm_gamma)
    if family == "autoencoder":
        return TorchAutoencoder(random_state=settings.random_state)
    raise ValueError(f"Unsupported anomaly model family: {settings.model_family!r}")


def fit_models(
    features: pl.DataFrame,
    train_end: date,
    settings: ModelSettings,
    selected_features: list[str] | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit models strictly on data at or before ``train_end``."""
    if settings.anomaly_weight < 0 or settings.peer_weight < 0:
        raise ValueError("Risk-score weights must be non-negative.")
    weight_total = settings.anomaly_weight + settings.peer_weight
    if weight_total == 0:
        raise ValueError("At least one risk-score weight must be positive.")
    raw_features = raw_feature_columns(features)
    z_features = [column for column in features.columns if column.startswith("z_")]
    train = features.filter(
        (pl.col("day") <= train_end) & pl.col("baseline_ready")
    )
    if train.height < 100:
        raise ValueError("Training period has fewer than 100 baseline-ready user-days.")
    # Constant features have undefined z-scores and carry no signal for Isolation Forest.
    available_model_features = [
        column
        for column in [*raw_features, *z_features]
        if train.get_column(column).null_count() < train.height
    ]
    if selected_features is None:
        model_features = available_model_features
    else:
        missing = set(selected_features).difference(available_model_features)
        if missing:
            raise ValueError(f"Selected features unavailable in train fold: {sorted(missing)}")
        model_features = selected_features
    if not model_features:
        raise ValueError("No usable anomaly-model features remain after selection.")

    user_profiles = train.group_by("user").agg(
        [pl.col(feature).mean().alias(feature) for feature in raw_features]
    )
    clusters = min(settings.peer_clusters, user_profiles.height)
    if clusters < 2:
        raise ValueError("At least two users with enough behavioural history are required.")

    peer_preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    profile_matrix = peer_preprocessor.fit_transform(_matrix(user_profiles, raw_features))
    peer_model = KMeans(n_clusters=clusters, random_state=settings.random_state, n_init="auto")
    profile_labels = peer_model.fit_predict(profile_matrix)
    user_clusters = dict(zip(user_profiles["user"].to_list(), profile_labels.tolist(), strict=True))

    org_centers: dict[str, list[float]] = {}
    if "org_peer_key" in train.columns:
        org_profiles = (
            train.filter(pl.col("org_peer_key").is_not_null())
            .group_by("org_peer_key")
            .agg(
                pl.col("user").n_unique().alias("_peer_users"),
                *[pl.col(feature).mean().alias(feature) for feature in raw_features],
            )
            .filter(pl.col("_peer_users") >= 5)
        )
        if org_profiles.height:
            org_profile_matrix = peer_preprocessor.transform(
                _matrix(org_profiles, raw_features)
            )
            org_centers = dict(
                zip(
                    org_profiles["org_peer_key"].to_list(),
                    org_profile_matrix.tolist(),
                    strict=True,
                )
            )

    # A robust scaler limits the influence of the most extreme raw activity counts.
    anomaly_preprocessor = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", RobustScaler())]
    )
    anomaly_matrix = anomaly_preprocessor.fit_transform(_matrix(train, model_features))
    anomaly_model = make_anomaly_model(settings, train.height)
    anomaly_model.fit(anomaly_matrix)

    train_anomaly = -anomaly_model.score_samples(anomaly_matrix)
    train_peer_daily = peer_preprocessor.transform(_matrix(train, raw_features))
    train_clusters = np.array(
        [user_clusters.get(user, -1) for user in train["user"].to_list()]
    )
    fallback_clusters = peer_model.predict(train_peer_daily)
    train_clusters = np.where(train_clusters >= 0, train_clusters, fallback_clusters)
    train_distance = np.linalg.norm(
        train_peer_daily - peer_model.cluster_centers_[train_clusters], axis=1
    )
    if org_centers:
        train_org_keys = train["org_peer_key"].to_list()
        org_distance = np.array(
            [
                np.linalg.norm(train_peer_daily[index] - np.array(org_centers[key]))
                if key in org_centers
                else train_distance[index]
                for index, key in enumerate(train_org_keys)
            ]
        )
        train_distance = org_distance

    peer_artifact: dict[str, object] = {
        "raw_features": raw_features,
        "preprocessor": peer_preprocessor,
        "model": peer_model,
        "user_clusters": user_clusters,
        "org_centers": org_centers,
        "distance_scale": np.quantile(train_distance, [0.05, 0.99]).tolist(),
        "train_end": train_end.isoformat(),
    }
    anomaly_artifact: dict[str, object] = {
        "model_features": model_features,
        "z_features": z_features,
        "preprocessor": anomaly_preprocessor,
        "model": anomaly_model,
        "family": settings.model_family,
        "anomaly_scale": np.quantile(train_anomaly, [0.05, 0.99]).tolist(),
        "risk_weights": {
            "anomaly": settings.anomaly_weight / weight_total,
            "peer": settings.peer_weight / weight_total,
        },
        "train_end": train_end.isoformat(),
    }
    return peer_artifact, anomaly_artifact


def save_models(
    peer_artifact: dict[str, object], anomaly_artifact: dict[str, object], model_dir: Path
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(peer_artifact, model_dir / "peer_group_kmeans.joblib")
    joblib.dump(anomaly_artifact, model_dir / "anomaly_model.joblib")
    # Compatibility alias for existing offline runs and the original README.
    joblib.dump(anomaly_artifact, model_dir / "isolation_forest.joblib")


def load_models(model_dir: Path) -> tuple[dict[str, object], dict[str, object]]:
    anomaly_path = model_dir / "anomaly_model.joblib"
    return (
        joblib.load(model_dir / "peer_group_kmeans.joblib"),
        joblib.load(anomaly_path if anomaly_path.is_file() else model_dir / "isolation_forest.joblib"),
    )


def _scaled(value: np.ndarray, limits: object) -> np.ndarray:
    low, high = (float(item) for item in limits)  # type: ignore[arg-type]
    if high <= low:
        return np.full_like(value, 0.5)
    # Keep rank information for extreme observations. Hard clipping at the 99th percentile
    # produced many identical 100-point scores, which invalidates Precision@K ranking.
    midpoint = (low + high) / 2
    scale = (high - low) / 2
    return 0.5 + np.arctan((value - midpoint) / scale) / np.pi


def score_features(
    features: pl.DataFrame,
    peer_artifact: dict[str, object],
    anomaly_artifact: dict[str, object],
) -> pl.DataFrame:
    """Score feature rows; unready baselines are retained but receive null risk scores."""
    output = features.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("risk_score"),
        pl.lit(None, dtype=pl.Float64).alias("anomaly_component"),
        pl.lit(None, dtype=pl.Float64).alias("peer_component"),
        pl.lit(None, dtype=pl.Float64).alias("peer_distance"),
        pl.lit(None, dtype=pl.Utf8).alias("peer_source"),
        pl.lit(None, dtype=pl.Utf8).alias("top_deviation_feature"),
        pl.lit(None, dtype=pl.Float64).alias("top_deviation_z"),
    )
    ready_positions = np.flatnonzero(features["baseline_ready"].to_numpy())
    if ready_positions.size == 0:
        return output

    ready = features.filter(pl.col("baseline_ready"))
    raw_features = peer_artifact["raw_features"]  # type: ignore[assignment]
    model_features = anomaly_artifact["model_features"]  # type: ignore[assignment]
    z_features = anomaly_artifact["z_features"]  # type: ignore[assignment]
    peer_preprocessor = peer_artifact["preprocessor"]
    peer_model = peer_artifact["model"]
    anomaly_preprocessor = anomaly_artifact["preprocessor"]
    anomaly_model = anomaly_artifact["model"]

    peer_matrix = peer_preprocessor.transform(_matrix(ready, raw_features))
    cluster_map = peer_artifact["user_clusters"]  # type: ignore[assignment]
    stored_clusters = np.array(
        [cluster_map.get(user, -1) for user in ready["user"].to_list()]
    )
    inferred_clusters = peer_model.predict(peer_matrix)
    cluster_ids = np.where(stored_clusters >= 0, stored_clusters, inferred_clusters)
    distances = np.linalg.norm(peer_matrix - peer_model.cluster_centers_[cluster_ids], axis=1)
    peer_sources = ["behavioral_cluster"] * ready.height
    org_centers = peer_artifact.get("org_centers", {})
    if org_centers and "org_peer_key" in ready.columns:
        for index, key in enumerate(ready["org_peer_key"].to_list()):
            if key in org_centers:
                distances[index] = np.linalg.norm(
                    peer_matrix[index] - np.array(org_centers[key])
                )
                peer_sources[index] = "ldap_role_department"

    anomaly_matrix = anomaly_preprocessor.transform(_matrix(ready, model_features))
    anomaly_values = -anomaly_model.score_samples(anomaly_matrix)
    anomaly_component = _scaled(anomaly_values, anomaly_artifact["anomaly_scale"])
    peer_component = _scaled(distances, peer_artifact["distance_scale"])
    risk_weights = anomaly_artifact.get("risk_weights", {"anomaly": 0.7, "peer": 0.3})
    risk = 100 * (
        float(risk_weights["anomaly"]) * anomaly_component
        + float(risk_weights["peer"]) * peer_component
    )

    z_matrix = _matrix(ready.fill_null(0), z_features).astype(float)
    max_positions = np.abs(z_matrix).argmax(axis=1)
    top_features = [z_features[position].removeprefix("z_") for position in max_positions]
    top_z = z_matrix[np.arange(z_matrix.shape[0]), max_positions]

    score_table = pl.DataFrame(
        {
            "_row_id": ready_positions,
            "risk_score": risk,
            "anomaly_component": anomaly_component,
            "peer_component": peer_component,
            "peer_distance": distances,
            "peer_source": peer_sources,
            "top_deviation_feature": top_features,
            "top_deviation_z": top_z,
        }
    )
    return (
        output.with_row_index("_row_id")
        .join(score_table, on="_row_id", how="left", suffix="_new")
        .with_columns(
            pl.coalesce(pl.col(f"{column}_new"), pl.col(column)).alias(column)
            for column in score_table.columns
            if column != "_row_id"
        )
        .drop(["_row_id", *[f"{column}_new" for column in score_table.columns if column != "_row_id"]])
    )
