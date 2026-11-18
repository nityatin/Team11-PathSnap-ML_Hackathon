
from __future__ import annotations

import json
import logging
import time
import warnings

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
)


# -------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "training_features"
    / "training_features.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "model_comparison"
)

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "models"

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

RANDOM_STATE = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("PathSnap.ModelTraining")


# -------------------------------------------------------
# MODEL DEFINITIONS
# -------------------------------------------------------

def make_models():

    models = {}

    # 1. Logistic Regression
    models["Logistic Regression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=200,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # 2. Decision Tree
    models["Decision Tree"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            DecisionTreeClassifier(
                max_depth=16,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # 3. Random Forest
    models["Random Forest"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=60,
                max_depth=16,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # 4. Extra Trees
    models["Extra Trees"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            ExtraTreesClassifier(
                n_estimators=60,
                max_depth=16,
                min_samples_leaf=10,
                max_features="sqrt",
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    # 5. Histogram Gradient Boosting
    models["HistGradientBoosting"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        (
            "classifier",
            HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=50,
                l2_regularization=1.0,
                early_stopping=True,
                random_state=RANDOM_STATE,
            ),
        ),
    ])

    return models


# -------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------

def load_training_data():

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(
            f"Feature dataset not found: {INPUT_FILE}"
        )

    logger.info("Loading training features...")

    required = [
        GROUP,
        "gps_index",
        "arc_id",
        TARGET,
        *FEATURES,
    ]

    df = pd.read_csv(
        INPUT_FILE,
        usecols=required,
        dtype={GROUP: "string"},
    )

    if df.empty:
        raise ValueError("Training feature dataset is empty.")

    if df[GROUP].isna().any():
        raise ValueError("Missing trip IDs found.")

    if df[["gps_index", "arc_id"]].isna().any().any():
        raise ValueError("Missing candidate identifiers found.")

    for column in FEATURES + [TARGET]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if df[FEATURES].isna().all(axis=1).any():
        raise ValueError(
            "Rows with all model features missing were found."
        )

    if not df[TARGET].isin([0, 1]).all():
        raise ValueError("Target contains labels other than 0 and 1.")

    # Features only. Never include reference metadata.
    X = df[FEATURES].astype(np.float32)

    y = df[TARGET].astype(np.int8)

    groups = df[GROUP].astype(str)

    if y.nunique() != 2:
        raise ValueError(
            "Both positive and negative labels are required."
        )

    if df[GROUP].nunique() < 2:
        raise ValueError(
            "At least two independent trips are required."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[GROUP, "gps_index", "arc_id"]
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{duplicate_count} duplicate candidate records found. "
            "Resolve duplicates before model training."
        )

    logger.info("Candidate rows: %s", f"{len(df):,}")
    logger.info("GPS observations: %s", f"{df.groupby([GROUP, 'gps_index']).ngroups:,}")
    logger.info("Training trips: %d", groups.nunique())

    logger.info(
        "Positive candidates: %s (%.2f%%)",
        f"{int(y.sum()):,}",
        y.mean() * 100,
    )

    logger.info(
        "Negative candidates: %s (%.2f%%)",
        f"{int((y == 0).sum()):,}",
        (y == 0).mean() * 100,
    )

    return df, X, y, groups


# -------------------------------------------------------
# EVALUATION METRICS
# -------------------------------------------------------

def calculate_metrics(
    y_true,
    probabilities,
    metadata,
):

    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= 0.5).astype(int)

    candidate_macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    candidate_precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    candidate_recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    candidate_accuracy = accuracy_score(
        y_true,
        predictions,
    )

    candidate_log_loss = log_loss(
        y_true,
        np.column_stack([
            1 - probabilities,
            probabilities,
        ]),
        labels=[0, 1],
    )

    # Rank candidates independently for each GPS observation.
    ranked = metadata[
        [GROUP, "gps_index"]
    ].copy()

    ranked["actual"] = np.asarray(y_true)
    ranked["probability"] = probabilities

    best_indices = (
        ranked.groupby(
            [GROUP, "gps_index"],
            sort=False,
        )["probability"]
        .idxmax()
    )

    selected = ranked.loc[best_indices]

    observation_top1_accuracy = float(
        selected["actual"].mean()
    )

    return {
        "candidate_accuracy": candidate_accuracy,
        "candidate_macro_f1": candidate_macro_f1,
        "candidate_precision": candidate_precision,
        "candidate_recall": candidate_recall,
        "candidate_log_loss": candidate_log_loss,
        "observation_top1_accuracy": observation_top1_accuracy,
        "evaluated_observations": len(selected),
    }


# -------------------------------------------------------
# CROSS VALIDATION
# -------------------------------------------------------

def run_cross_validation(
    df,
    X,
    y,
    groups,
    number_of_folds,
):

    unique_groups = groups.nunique()

    if number_of_folds < 2:
        raise ValueError("At least two folds are required.")

    if number_of_folds > unique_groups:
        raise ValueError(
            "Number of folds cannot exceed number of trips."
        )

    splitter = GroupKFold(
        n_splits=number_of_folds
    )

    models = make_models()

    fold_results = []
    all_model_predictions = {}

    best_name = None
    best_score = -np.inf
    best_macro_f1 = -np.inf

    for model_name, original_model in models.items():

        logger.info("\n" + "=" * 60)
        logger.info("MODEL: %s", model_name)
        logger.info("=" * 60)

        oof_probabilities = np.full(
            len(df),
            np.nan,
            dtype=np.float32,
        )

        total_fit_time = 0.0
        total_inference_time = 0.0

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(X, y, groups),
            start=1,
        ):

            logger.info(
                "Fold %d/%d | Train rows: %s | Validation rows: %s",
                fold,
                number_of_folds,
                f"{len(train_idx):,}",
                f"{len(val_idx):,}",
            )

            train_groups = set(
                groups.iloc[train_idx]
            )

            validation_groups = set(
                groups.iloc[val_idx]
            )

            if train_groups.intersection(validation_groups):
                raise AssertionError(
                    "Trip leakage detected between training and validation."
                )

            model = clone(original_model)

            X_train = X.iloc[train_idx]
            X_val = X.iloc[val_idx]

            y_train = y.iloc[train_idx]
            y_val = y.iloc[val_idx]

            # HistGradientBoosting receives explicit
            # balanced sample weights.
            fit_kwargs = {}

            if model_name == "HistGradientBoosting":

                weights = compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )

                fit_kwargs[
                    "classifier__sample_weight"
                ] = weights

            fit_start = time.perf_counter()

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")

                model.fit(
                    X_train,
                    y_train,
                    **fit_kwargs,
                )

                for warning in caught:
                    logger.warning(
                        "%s: %s",
                        model_name,
                        str(warning.message),
                    )

            fit_seconds = (
                time.perf_counter() - fit_start
            )

            total_fit_time += fit_seconds

            logger.info(
                "Training time: %.2f seconds",
                fit_seconds,
            )

            predict_start = time.perf_counter()

            probabilities = model.predict_proba(
                X_val
            )[:, 1]

            inference_seconds = (
                time.perf_counter() - predict_start
            )

            total_inference_time += inference_seconds

            oof_probabilities[val_idx] = probabilities

            metadata = df.iloc[val_idx]

            metrics = calculate_metrics(
                y_val.to_numpy(),
                probabilities,
                metadata,
            )

            result = {
                "model": model_name,
                "fold": fold,
                "training_rows": len(train_idx),
                "validation_rows": len(val_idx),
                "training_trips": len(train_groups),
                "validation_trips": len(validation_groups),
                "training_time_seconds": fit_seconds,
                "inference_time_seconds": inference_seconds,
                **metrics,
            }

            fold_results.append(result)

            logger.info(
                "Candidate Macro F1: %.4f | "
                "Observation Top-1: %.4f | "
                "Log Loss: %.4f",
                metrics["candidate_macro_f1"],
                metrics["observation_top1_accuracy"],
                metrics["candidate_log_loss"],
            )

            del model

        if np.isnan(oof_probabilities).any():
            raise RuntimeError(
                f"Incomplete out-of-fold predictions for {model_name}"
            )

        # Aggregate out-of-fold predictions.
        oof_metrics = calculate_metrics(
            y.to_numpy(),
            oof_probabilities,
            df,
        )

        all_model_predictions[model_name] = (
            oof_probabilities
        )

        logger.info("\nOVERALL OOF RESULTS: %s", model_name)

        for metric, value in oof_metrics.items():
            logger.info("%s: %s", metric, value)

        # Select primarily on observation Top-1 accuracy,
        # using Macro F1 as the tie-breaker.
        score = oof_metrics["observation_top1_accuracy"]
        macro_f1 = oof_metrics["candidate_macro_f1"]

        if (
            score > best_score
            or (
                np.isclose(score, best_score)
                and macro_f1 > best_macro_f1
            )
        ):

            best_score = score
            best_macro_f1 = macro_f1
            best_name = model_name

        logger.info(
            "Completed %s. Total training: %.2f seconds",
            model_name,
            total_fit_time,
        )

    return (
        fold_results,
        all_model_predictions,
        best_name,
    )


# -------------------------------------------------------
# FINAL MODEL AND REPORTS
# -------------------------------------------------------

def save_results(
    df,
    X,
    y,
    groups,
    fold_results,
    predictions,
    best_name,
    number_of_folds,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Per-fold results
    fold_df = pd.DataFrame(fold_results)

    fold_df.to_csv(
        OUTPUT_DIR / "cv_fold_metrics.csv",
        index=False,
    )

    # Overall model comparison
    comparison = []

    for model_name, probabilities in predictions.items():

        metrics = calculate_metrics(
            y.to_numpy(),
            probabilities,
            df,
        )

        model_folds = fold_df[
            fold_df["model"] == model_name
        ]

        comparison.append({
            "model": model_name,
            **metrics,
            "mean_cv_macro_f1": model_folds[
                "candidate_macro_f1"
            ].mean(),
            "std_cv_macro_f1": model_folds[
                "candidate_macro_f1"
            ].std(),
            "mean_fold_top1_accuracy": model_folds[
                "observation_top1_accuracy"
            ].mean(),
            "total_training_time_seconds": model_folds[
                "training_time_seconds"
            ].sum(),
            "total_inference_time_seconds": model_folds[
                "inference_time_seconds"
            ].sum(),
        })

    comparison_df = pd.DataFrame(comparison)

    comparison_df = comparison_df.sort_values(
        by=[
            "observation_top1_accuracy",
            "candidate_macro_f1",
        ],
        ascending=False,
    )

    comparison_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    # Save out-of-fold predictions for reproducibility.
    oof = df[
        [GROUP, "gps_index", "arc_id", TARGET]
    ].copy()

    for model_name, probabilities in predictions.items():

        safe_name = (
            model_name.lower()
            .replace(" ", "_")
        )

        oof[f"{safe_name}_probability"] = probabilities

    oof.to_csv(
        OUTPUT_DIR / "oof_predictions.csv.gz",
        index=False,
        compression="gzip",
    )

    # ---------------------------------------------------
    # FIT SELECTED MODEL USING ALL TRAINING DATA
    # ---------------------------------------------------

    logger.info(
        "Refitting selected model on all accepted training data: %s",
        best_name,
    )

    final_model = clone(
        make_models()[best_name]
    )

    fit_kwargs = {}

    if best_name == "HistGradientBoosting":

        fit_kwargs[
            "classifier__sample_weight"
        ] = compute_sample_weight(
            class_weight="balanced",
            y=y,
        )

    final_start = time.perf_counter()

    final_model.fit(
        X,
        y,
        **fit_kwargs,
    )

    final_training_time = (
        time.perf_counter() - final_start
    )

    model_path = (
        ARTIFACT_DIR / "best_candidate_ranker.joblib"
    )

    joblib.dump(
        final_model,
        model_path,
        compress=3,
    )

    metadata = {
        "selected_model": best_name,
        "selection_metric": (
            "Observation-level Top-1 Accuracy, "
            "with candidate Macro F1 as tie-breaker"
        ),
        "cross_validation": {
            "method": "GroupKFold",
            "number_of_folds": number_of_folds,
            "group_column": GROUP,
            "group_overlap_allowed": False,
        },
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "training_trips": int(groups.nunique()),
        "training_candidate_rows": len(df),
        "final_model_training_time_seconds": (
            final_training_time
        ),
        "artifact": str(model_path),
    }

    with (
        OUTPUT_DIR / "model_selection.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("\n" + "=" * 65)
    logger.info("MODEL COMPARISON COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "\n%s",
        comparison_df.to_string(index=False),
    )

    logger.info("\nSELECTED MODEL: %s", best_name)
    logger.info("Saved final model: %s", model_path)


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    logger.info("Starting PathSnap model experimentation.")

    df, X, y, groups = load_training_data()

    folds = min(
        args.folds,
        groups.nunique(),
    )

    if folds < 2:
        raise ValueError(
            "Insufficient training trips for GroupKFold."
        )

    logger.info("Using %d GroupKFold folds.", folds)

    (
        fold_results,
        predictions,
        best_name,
    ) = run_cross_validation(
        df,
        X,
        y,
        groups,
        folds,
    )

    save_results(
        df,
        X,
        y,
        groups,
        fold_results,
        predictions,
        best_name,
        folds,
    )


if __name__ == "__main__":
    main()
