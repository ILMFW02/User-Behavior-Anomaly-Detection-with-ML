"""Command-line entry points for the offline UEBA workflow."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl

from .baseline import add_causal_baselines
from .config import Settings, load_settings
from .dataset import format_manifest, inspect_dataset
from .evaluation import evaluate_scores
from .experiments import run_model_families
from .features import build_user_day_features
from .labels import build_insider_user_day_labels
from .models import fit_models, load_models, save_models, score_features
from .reproducibility import write_run_manifest
from .temporal import folds_from_frame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline CERT UEBA pipeline")
    parser.add_argument("--config", default="config.toml", help="Path to project TOML configuration")
    parser.add_argument(
        "--hash-inputs",
        action="store_true",
        help="Calculate full SHA-256 for input files in the run manifest (slow for raw logs).",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate-data", help="Inspect CERT filenames and CSV headers")
    labels = commands.add_parser("build-labels", help="Create held-out user-day labels from CERT answers")
    labels.add_argument("--release", default="r4.2", help="CERT release label, e.g. r4.2 or r6.2")
    labels.add_argument(
        "--early-warning-days",
        type=int,
        default=0,
        help="Label preceding days only; keep event-day detection labels as a separate task.",
    )
    commands.add_parser("build-features", help="Create user-day and baseline feature parquet files")

    train = commands.add_parser("train", help="Fit peer grouping and anomaly models")
    train.add_argument(
        "--train-end",
        required=True,
        type=date.fromisoformat,
        help="Last inclusive training date, formatted YYYY-MM-DD",
    )

    commands.add_parser("score", help="Score all baseline-ready user-days")
    evaluate = commands.add_parser("evaluate", help="Evaluate a scored file against held-out labels")
    evaluate.add_argument(
        "--labels",
        type=Path,
        help="Optional CSV with user,day,label columns; defaults to generated CERT labels",
    )
    evaluate.add_argument(
        "--test-start",
        required=True,
        type=date.fromisoformat,
        help="First inclusive held-out test date, formatted YYYY-MM-DD",
    )
    evaluate.add_argument(
        "--alert-k",
        type=int,
        help="Fixed analyst alert budget; defaults to [evaluation].alert_k.",
    )
    cross_validate = commands.add_parser(
        "cross-validate", help="Run feasible expanding chronological folds and model baselines"
    )
    cross_validate.add_argument(
        "--labels", type=Path, help="CSV with user,day,label; defaults to generated CERT labels"
    )
    cross_validate.add_argument(
        "--model-families",
        default="isolation_forest,lof,ocsvm",
        help="Comma-separated detector families; autoencoder is optional.",
    )
    return parser


def _model_features_path(settings: Settings) -> Path:
    return settings.paths.processed_dir / "user_day_model_features.parquet"


def _scored_path(settings: Settings) -> Path:
    return settings.paths.report_dir / "scored_user_days.parquet"


def _labels_path(settings: Settings) -> Path:
    return settings.paths.processed_dir / "insider_user_day_labels.csv"


def _build_labels(settings: Settings, release: str, early_warning_days: int) -> None:
    labels = build_insider_user_day_labels(
        settings.paths.answers_dir, release=release, early_warning_days=early_warning_days
    )
    settings.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    labels.write_csv(_labels_path(settings))
    print(f"Wrote {labels.height:,} positive insider user-days to {_labels_path(settings)}")


def _build_features(settings: Settings) -> None:
    features = build_user_day_features(settings.paths.raw_dir, settings.features)
    model_features, _ = add_causal_baselines(features, settings.baseline)
    settings.paths.processed_dir.mkdir(parents=True, exist_ok=True)
    features.write_parquet(settings.paths.processed_dir / "user_day_features.parquet")
    model_features.write_parquet(_model_features_path(settings))
    print(
        f"Wrote {features.height:,} user-day rows and "
        f"{len(features.columns) - 2} raw features to {settings.paths.processed_dir}"
    )


def _train(settings: Settings, train_end: date) -> None:
    path = _model_features_path(settings)
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist. Run `ueba build-features` first.")
    peer_artifact, anomaly_artifact = fit_models(
        pl.read_parquet(path), train_end, settings.model
    )
    save_models(peer_artifact, anomaly_artifact, settings.paths.model_dir)
    print(f"Saved models trained through {train_end} to {settings.paths.model_dir}")


def _score(settings: Settings) -> None:
    features_path = _model_features_path(settings)
    if not features_path.is_file():
        raise FileNotFoundError(f"{features_path} does not exist. Run `ueba build-features` first.")
    peer_artifact, anomaly_artifact = load_models(settings.paths.model_dir)
    scored = score_features(pl.read_parquet(features_path), peer_artifact, anomaly_artifact)
    settings.paths.report_dir.mkdir(parents=True, exist_ok=True)
    scored.write_parquet(_scored_path(settings))
    highest = scored.filter(pl.col("risk_score").is_not_null()).sort(
        "risk_score", descending=True
    ).head(10)
    print(f"Wrote scored user-days to {_scored_path(settings)}")
    print(
        highest.select(["user", "day", "risk_score", "top_deviation_feature"]).to_dicts()
    )


def _evaluate(settings: Settings, labels: Path, test_start: date, alert_k: int) -> None:
    score_path = _scored_path(settings)
    if not score_path.is_file():
        raise FileNotFoundError(f"{score_path} does not exist. Run `ueba score` first.")
    metrics = evaluate_scores(pl.read_parquet(score_path), labels, test_start, alert_k)
    settings.paths.report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = settings.paths.report_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


def _cross_validate(settings: Settings, labels: Path, families: tuple[str, ...]) -> None:
    features_path = _model_features_path(settings)
    if not features_path.is_file():
        raise FileNotFoundError(f"{features_path} does not exist. Run `ueba build-features` first.")
    features = pl.read_parquet(features_path)
    folds = folds_from_frame(
        features,
        initial_train_days=settings.temporal.initial_train_days,
        validation_days=settings.temporal.validation_days,
        test_days=settings.temporal.test_days,
        step_days=settings.temporal.step_days,
        embargo_days=settings.temporal.embargo_days,
    )
    if not folds:
        raise ValueError(
            "No feasible temporal fold. Reduce temporal windows or run a dataset with a longer timeline."
        )
    results = [
        run_model_families(
            features,
            labels,
            fold,
            settings.model,
            settings.selection,
            families=families,
            alert_k=settings.evaluation.alert_k,
        )
        for fold in folds
    ]
    output = pl.concat(results, how="diagonal_relaxed")
    settings.paths.report_dir.mkdir(parents=True, exist_ok=True)
    path = settings.paths.report_dir / "cross_validation_metrics.csv"
    output.write_csv(path)
    print(f"Wrote {output.height} fold/model metrics to {path}")


def _manifest(settings: Settings, args: argparse.Namespace, inputs: list[Path]) -> None:
    path = write_run_manifest(
        settings.paths.report_dir,
        command=args.command,
        config_path=Path(args.config).resolve(),
        inputs=inputs,
        seed=settings.model.random_state,
        hash_inputs=args.hash_inputs,
    )
    print(f"Wrote reproducibility manifest to {path}")


def main() -> None:
    args = _parser().parse_args()
    settings = load_settings(args.config)
    if args.command == "validate-data":
        print(format_manifest(inspect_dataset(settings.paths.raw_dir)))
    elif args.command == "build-labels":
        _build_labels(settings, args.release, args.early_warning_days)
    elif args.command == "build-features":
        _build_features(settings)
    elif args.command == "train":
        _train(settings, args.train_end)
    elif args.command == "score":
        _score(settings)
    elif args.command == "evaluate":
        _evaluate(
            settings,
            args.labels or _labels_path(settings),
            args.test_start,
            args.alert_k or settings.evaluation.alert_k,
        )
    elif args.command == "cross-validate":
        _cross_validate(
            settings,
            args.labels or _labels_path(settings),
            tuple(item.strip() for item in args.model_families.split(",") if item.strip()),
        )
    _manifest(
        settings,
        args,
        [
            _model_features_path(settings),
            _scored_path(settings),
            _labels_path(settings),
        ],
    )


if __name__ == "__main__":
    main()
