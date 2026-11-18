#!/usr/bin/env python3
"""Bayesian hyperparameter optimisation for PathSnap's Random Forest ranker.

Uses only the accepted training feature dataset. Validation is grouped by trip_id
so rows from one trip cannot cross from a fold's training partition to validation.
The optimisation objective is observation-level Top-1 accuracy: for each GPS
observation, select the candidate with the highest positive-class probability.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "training_features"
    / "training_features.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "bayesian_trials"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"

FEATURES = [
    "distance_m",
    "bearing_difference_deg",
    "road_length_m",
    "projection_fraction",
    "gps_speed_mps",
    "time_delta_s",
]
TARGET = "is_reference"
GROUP = "trip_id"
OBSERVATION_ID = "gps_index"
RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PathSnap.BayesianOptimization")


def load_training_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load training features only; never read student/evaluator route labels."""
    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Training feature dataset not found: {INPUT_FILE}\n"
            "Run build_training_features.py first."
        )

    required = FEATURES + [TARGET, GROUP, OBSERVATION_ID]
    logger.info("Loading training features from %s", INPUT_FILE)
    df = pd.read_csv(INPUT_FILE, usecols=required, dtype={GROUP: "string"})

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df.empty:
        raise ValueError("Training feature dataset is empty.")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype("int8")
    df[OBSERVATION_ID] = pd.to_numeric(
        df[OBSERVATION_ID], errors="raise"
    ).astype("int64")
    df[FEATURES] = df[FEATURES].apply(pd.to_numeric, errors="coerce")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            f"Expected binary labels 0 and 1; found {sorted(df[TARGET].unique())}"
        )
    if df[GROUP].isna().any() or df[OBSERVATION_ID].isna().any():
        raise ValueError("trip_id/gps_index contains missing values.")

    # Each GPS observation must have at least one positive candidate.
    positive_per_observation = (
        df.groupby([GROUP, OBSERVATION_ID], sort=False)[TARGET].sum()
    )
    invalid_observations = int((positive_per_observation < 1).sum())
    if invalid_observations:
        raise ValueError(
            f"Found {invalid_observations} observations without a positive candidate."
        )

    X = df[FEATURES]
    y = df[TARGET]
    groups = df[GROUP]

    logger.info(
        "Loaded %s candidate rows across %s training trips and %s GPS observations.",
        f"{len(df):,}",
        groups.nunique(),
        f"{positive_per_observation.size:,}",
    )
    logger.info(
        "Positive candidate share: %.2f%%",
        100.0 * float(y.mean()),
    )
    return df, X, y, groups


def make_pipeline(params: dict[str, Any]) -> Pipeline:
    """Build the same median-imputation + Random Forest structure as baseline."""
    classifier = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        min_samples_split=int(params["min_samples_split"]),
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        max_samples=float(params["max_samples"]),
        bootstrap=True,
        class_weight=str(params["class_weight"]),
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", classifier),
        ]
    )


def baseline_params() -> dict[str, Any]:
    """Current Random Forest baseline from train_and_compare_models.py."""
    return {
        "n_estimators": 60,
        "max_depth": 16,
        "min_samples_split": 2,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "max_samples": 1.0,
        "class_weight": "balanced_subsample",
    }


def observation_top1(
    validation_rows: pd.DataFrame,
    probabilities: np.ndarray,
) -> tuple[float, int]:
    """Return pooled Top-1 accuracy and number of GPS observations."""
    scored = validation_rows[[GROUP, OBSERVATION_ID, TARGET]].copy()
    scored["_positive_probability"] = probabilities

    # One highest-probability candidate per GPS observation.
    best_indices = scored.groupby(
        [GROUP, OBSERVATION_ID], sort=False
    )["_positive_probability"].idxmax()
    selected_labels = scored.loc[best_indices, TARGET].to_numpy(dtype=np.int8)

    if selected_labels.size == 0:
        raise ValueError("Validation fold contains no GPS observations.")
    return float(selected_labels.mean()), int(selected_labels.size)


def evaluate_params(
    df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    params: dict[str, Any],
    n_splits: int,
    label: str,
) -> dict[str, Any]:
    """Evaluate one parameter set using fixed GroupKFold partitions."""
    splitter = GroupKFold(n_splits=n_splits)
    fold_rows: list[dict[str, Any]] = []
    correct_observations = 0.0
    total_observations = 0
    fold_fit_seconds = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        splitter.split(X, y, groups), start=1
    ):
        train_groups = set(groups.iloc[train_idx].unique())
        valid_groups = set(groups.iloc[valid_idx].unique())
        overlap = train_groups.intersection(valid_groups)
        if overlap:
            raise AssertionError(
                f"Trip leakage in fold {fold_number}: {sorted(overlap)[:5]}"
            )

        model = make_pipeline(params)
        fit_start = time.perf_counter()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        fit_seconds = time.perf_counter() - fit_start
        fold_fit_seconds.append(fit_seconds)

        # The positive class is encoded as 1.
        class_values = model.named_steps["classifier"].classes_
        positive_column = int(np.flatnonzero(class_values == 1)[0])
        probabilities = model.predict_proba(X.iloc[valid_idx])[:, positive_column]
        validation_rows = df.iloc[valid_idx]
        fold_top1, number_of_observations = observation_top1(
            validation_rows, probabilities
        )

        correct_observations += fold_top1 * number_of_observations
        total_observations += number_of_observations
        fold_rows.append(
            {
                "fold": fold_number,
                "top1_accuracy": fold_top1,
                "gps_observations": number_of_observations,
                "training_trips": len(train_groups),
                "validation_trips": len(valid_groups),
                "fit_seconds": fit_seconds,
            }
        )
        logger.info(
            "%s | fold %d/%d | Top-1 %.5f | observations %s | fit %.1fs",
            label,
            fold_number,
            n_splits,
            fold_top1,
            f"{number_of_observations:,}",
            fit_seconds,
        )
        del model, probabilities
        gc.collect()

    pooled_score = correct_observations / total_observations
    return {
        "observation_top1_accuracy": float(pooled_score),
        "evaluated_observations": int(total_observations),
        "fold_results": fold_rows,
        "total_fit_seconds": float(sum(fold_fit_seconds)),
    }


def suggest_params(trial: optuna.Trial) -> dict[str, Any]:
    max_features_choice = trial.suggest_categorical(
        "max_features_choice", ["sqrt", "log2", "0.5", "0.75"]
    )
    max_features: str | float
    if max_features_choice in {"sqrt", "log2"}:
        max_features = max_features_choice
    else:
        max_features = float(max_features_choice)

    return {
        "n_estimators": trial.suggest_int("n_estimators", 60, 140, step=20),
        "max_depth": trial.suggest_int("max_depth", 10, 24, step=2),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 20),
        "max_features": max_features,
        "max_samples": trial.suggest_float("max_samples", 0.65, 1.0),
        "class_weight": trial.suggest_categorical(
            "class_weight", ["balanced_subsample", "balanced"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune the PathSnap Random Forest using Optuna TPE and GroupKFold."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=12,
        help="Number of additional Bayesian optimisation trials (default: 12).",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="Number of GroupKFold folds during tuning (default: 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_STATE, help="Sampler random seed."
    )
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("--trials must be at least 1")
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df, X, y, groups = load_training_data()
    unique_trips = int(groups.nunique())
    n_splits = min(args.folds, unique_trips)
    if n_splits < 2:
        raise ValueError("At least two distinct training trips are required.")

    logger.info("Using %d-fold GroupKFold for Bayesian optimisation.", n_splits)
    logger.info("Objective: pooled observation-level Top-1 accuracy.")
    logger.info("Student-test and evaluator-test partitions are not loaded.")

    # Baseline is measured on exactly the same grouped folds for comparison.
    logger.info("Evaluating original Random Forest baseline on tuning folds...")
    baseline_result = evaluate_params(
        df,
        X,
        y,
        groups,
        baseline_params(),
        n_splits,
        "BASELINE RF",
    )

    database_path = OUTPUT_DIR / "random_forest_tpe_study.db"
    storage_url = f"sqlite:///{database_path.as_posix()}"
    study = optuna.create_study(
        study_name="pathsnap_random_forest_top1_tpe",
        storage=storage_url,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed, multivariate=True, n_startup_trials=5
        ),
        load_if_exists=True,
    )

    fold_indices = list(GroupKFold(n_splits=n_splits).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        logger.info("Starting TPE trial %d with params: %s", trial.number, params)

        fold_correct = 0.0
        fold_observations = 0
        fold_metrics = []

        for fold_number, (train_idx, valid_idx) in enumerate(
            fold_indices, start=1
        ):
            train_groups = set(groups.iloc[train_idx].unique())
            valid_groups = set(groups.iloc[valid_idx].unique())
            if train_groups.intersection(valid_groups):
                raise AssertionError("Trip leakage detected in optimisation fold.")

            model = make_pipeline(params)
            fit_start = time.perf_counter()
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
            fit_seconds = time.perf_counter() - fit_start

            class_values = model.named_steps["classifier"].classes_
            positive_column = int(np.flatnonzero(class_values == 1)[0])
            probabilities = model.predict_proba(X.iloc[valid_idx])[:, positive_column]
            top1, observation_count = observation_top1(
                df.iloc[valid_idx], probabilities
            )
            fold_correct += top1 * observation_count
            fold_observations += observation_count
            fold_metrics.append(
                {
                    "fold": fold_number,
                    "top1_accuracy": top1,
                    "gps_observations": observation_count,
                    "fit_seconds": fit_seconds,
                }
            )
            logger.info(
                "Trial %d | fold %d/%d | Top-1 %.5f | fit %.1fs",
                trial.number,
                fold_number,
                n_splits,
                top1,
                fit_seconds,
            )
            del model, probabilities
            gc.collect()

        score = fold_correct / fold_observations
        trial.set_user_attr("fold_metrics", fold_metrics)
        trial.set_user_attr("evaluated_observations", int(fold_observations))
        logger.info("Trial %d completed | pooled Top-1 %.6f", trial.number, score)
        return float(score)

    study.optimize(
        objective,
        n_trials=args.trials,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    completed_trials = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        raise RuntimeError("No optimisation trials completed successfully.")

    best_trial = study.best_trial
    best_params = dict(best_trial.params)
    # Convert the categorical string representation back to sklearn values.
    feature_choice = best_params.pop("max_features_choice")
    best_params["max_features"] = (
        feature_choice
        if feature_choice in {"sqrt", "log2"}
        else float(feature_choice)
    )

    best_cv_result = {
        "observation_top1_accuracy": float(best_trial.value),
        "evaluated_observations": int(
            best_trial.user_attrs.get("evaluated_observations", 0)
        ),
        "fold_results": best_trial.user_attrs.get("fold_metrics", []),
    }

    # Fit the optimised configuration on all accepted training observations.
    logger.info("Fitting best Random Forest on all accepted training rows...")
    final_model = make_pipeline(best_params)
    final_start = time.perf_counter()
    final_model.fit(X, y)
    final_fit_seconds = time.perf_counter() - final_start

    model_path = MODEL_DIR / "bayesian_optimized_random_forest.joblib"
    joblib.dump(final_model, model_path, compress=3)

    # Preserve the complete study and machine-readable experiment summary.
    trials_csv = OUTPUT_DIR / "random_forest_tpe_trials.csv"
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "state", "duration")
    )
    trials_df.to_csv(trials_csv, index=False)

    summary = {
        "study_name": study.study_name,
        "sampler": "Optuna TPESampler (Tree-structured Parzen Estimator)",
        "direction": "maximize",
        "objective": "pooled observation-level Top-1 accuracy",
        "training_data": str(INPUT_FILE.relative_to(PROJECT_ROOT)),
        "student_test_loaded": False,
        "evaluator_test_loaded": False,
        "group_column": GROUP,
        "number_of_training_trips": unique_trips,
        "number_of_candidate_rows": int(len(df)),
        "number_of_gps_observations": int(
            df.groupby([GROUP, OBSERVATION_ID], sort=False).ngroups
        ),
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "validation": {
            "method": "GroupKFold",
            "number_of_folds": n_splits,
            "trip_overlap_allowed": False,
        },
        "baseline_random_forest": {
            "parameters": baseline_params(),
            "grouped_cv_observation_top1_accuracy": baseline_result[
                "observation_top1_accuracy"
            ],
            "fold_results": baseline_result["fold_results"],
        },
        "optimisation": {
            "requested_additional_trials": args.trials,
            "total_completed_trials_in_study": len(completed_trials),
            "best_trial_number": int(best_trial.number),
            "best_grouped_cv_observation_top1_accuracy": float(best_trial.value),
            "best_parameters": best_params,
            "best_trial_fold_results": best_cv_result["fold_results"],
        },
        "comparison": {
            "baseline_top1": baseline_result["observation_top1_accuracy"],
            "optimised_top1": float(best_trial.value),
            "absolute_improvement_percentage_points": 100.0
            * (
                float(best_trial.value)
                - baseline_result["observation_top1_accuracy"]
            ),
        },
        "final_fit_seconds": final_fit_seconds,
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "trial_results_csv": str(trials_csv.relative_to(PROJECT_ROOT)),
        "study_database": str(database_path.relative_to(PROJECT_ROOT)),
        "random_state": args.seed,
    }

    summary_path = OUTPUT_DIR / "random_forest_optimization_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("\n%s", "=" * 68)
    logger.info("BAYESIAN OPTIMISATION COMPLETED")
    logger.info("Baseline grouped CV Top-1: %.6f", baseline_result["observation_top1_accuracy"])
    logger.info("Optimised grouped CV Top-1: %.6f", float(best_trial.value))
    logger.info(
        "Absolute improvement: %+.4f percentage points",
        summary["comparison"]["absolute_improvement_percentage_points"],
    )
    logger.info("Best parameters: %s", best_params)
    logger.info("Saved model: %s", model_path)
    logger.info("Saved trial table: %s", trials_csv)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved Optuna study database: %s", database_path)


if __name__ == "__main__":
    main()
