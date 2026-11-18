
from __future__ import annotations

import json
import logging
import re
import time

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree

from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

# Reuse the exact five model configurations used in CV.
from train_and_compare_models import (
    FEATURES,
    TARGET,
    make_models,
)


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

TRAIN_FEATURES = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "training_features"
    / "training_features.csv"
)

CV_RESULTS = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "model_comparison"
    / "model_comparison.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "student_testing"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "student_testing_evaluation"
)

CANDIDATE_RADIUS_M = 50.0
LABEL_DISTANCE_LIMIT_M = 50.0
EARTH_RADIUS_KM = 6371.0088

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("PathSnap.StudentTesting")


# ---------------------------------------------------------
# CONFIGURATION AND DATA LOADING
# ---------------------------------------------------------

def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml configuration.")

    return config


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()

    return path


def read_table(path: Path, columns: int, dtype=float):
    data = np.loadtxt(path, dtype=dtype)
    data = np.asarray(data)

    if data.size == 0:
        return np.empty((0, columns), dtype=dtype)

    return data.reshape(-1, columns)


def load_trip_without_labels(
    trip_dir: Path,
    trip_id: str,
):
    """
    Prediction-side loader.

    Deliberately loads only:
      - .track
      - .nodes
      - .arcs

    It MUST NOT open .route.
    """

    prefix = trip_dir / trip_id

    track = read_table(
        Path(str(prefix) + ".track"),
        3,
    )

    nodes = read_table(
        Path(str(prefix) + ".nodes"),
        2,
    )

    arcs = read_table(
        Path(str(prefix) + ".arcs"),
        2,
        dtype=np.int64,
    )

    return track, nodes, arcs


# ---------------------------------------------------------
# GEOGRAPHICAL PROCESSING
# ---------------------------------------------------------

def create_projection(nodes: np.ndarray):

    lon0 = float(np.median(nodes[:, 0]))
    lat0 = float(np.median(nodes[:, 1]))

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    return Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )


def angle_difference(a, b):

    return np.abs(
        (a - b + 180) % 360 - 180
    )


def calculate_gps_motion(
    gps_xy: np.ndarray,
    timestamps: np.ndarray,
):

    count = len(gps_xy)

    bearings = np.full(count, np.nan)
    speeds = np.full(count, np.nan)
    time_deltas = np.full(count, np.nan)

    for i in range(count):

        if count < 2:
            break

        if i < count - 1:
            start, end = i, i + 1
        else:
            start, end = i - 1, i

        dx, dy = (
            gps_xy[end] - gps_xy[start]
        )

        distance = float(np.hypot(dx, dy))

        dt = float(
            timestamps[end] - timestamps[start]
        )

        time_deltas[i] = dt

        if distance > 1e-6:
            bearings[i] = (
                np.degrees(np.arctan2(dx, dy)) + 360
            ) % 360

        if dt > 0:
            speeds[i] = distance / dt

    return bearings, speeds, time_deltas


# ---------------------------------------------------------
# PREDICTION-SIDE CANDIDATE GENERATION
# ---------------------------------------------------------

def generate_inference_candidates(
    trip_dir: Path,
    trip_id: str,
    radius_m: float,
):
    """
    Generates candidate features without accessing .route.
    """

    track, nodes, arcs = load_trip_without_labels(
        trip_dir,
        trip_id,
    )

    if not len(track) or not len(nodes) or not len(arcs):
        raise ValueError(
            f"Empty required input file in trip {trip_id}"
        )

    valid_arcs = (
        (arcs[:, 0] >= 0)
        & (arcs[:, 0] < len(nodes))
        & (arcs[:, 1] >= 0)
        & (arcs[:, 1] < len(nodes))
    )

    original_arc_ids = np.flatnonzero(valid_arcs)
    arc_pairs = arcs[valid_arcs]

    if not len(arc_pairs):
        raise ValueError(
            f"No valid arcs in trip {trip_id}"
        )

    transformer = create_projection(nodes)

    node_x, node_y = transformer.transform(
        nodes[:, 0],
        nodes[:, 1],
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0],
        track[:, 1],
    )

    node_xy = np.column_stack([
        node_x,
        node_y,
    ])

    gps_xy = np.column_stack([
        gps_x,
        gps_y,
    ])

    arc_starts = node_xy[arc_pairs[:, 0]]
    arc_ends = node_xy[arc_pairs[:, 1]]

    arc_vectors = arc_ends - arc_starts

    arc_lengths = np.linalg.norm(
        arc_vectors,
        axis=1,
    )

    arc_bearings = (
        np.degrees(
            np.arctan2(
                arc_vectors[:, 0],
                arc_vectors[:, 1],
            )
        ) + 360
    ) % 360

    geometries = [
        LineString([start, end])
        for start, end in zip(
            arc_starts,
            arc_ends,
        )
    ]

    tree = STRtree(geometries)

    gps_bearings, gps_speeds, time_deltas = (
        calculate_gps_motion(
            gps_xy,
            track[:, 2],
        )
    )

    records = []

    observations_with_candidates = 0

    for gps_index, (x, y) in enumerate(gps_xy):

        point = Point(float(x), float(y))

        search_area = box(
            x - radius_m,
            y - radius_m,
            x + radius_m,
            y + radius_m,
        )

        local_indices = np.asarray(
            tree.query(search_area),
            dtype=int,
        )

        if not len(local_indices):
            continue

        local_distances = np.asarray([
            geometries[index].distance(point)
            for index in local_indices
        ])

        selected = local_distances <= radius_m

        local_indices = local_indices[selected]
        local_distances = local_distances[selected]

        if not len(local_indices):
            continue

        observations_with_candidates += 1

        gps_bearing = gps_bearings[gps_index]

        for local_index, distance in zip(
            local_indices,
            local_distances,
        ):

            start_node, end_node = arc_pairs[local_index]

            vector = arc_vectors[local_index]

            denominator = float(
                np.dot(vector, vector)
            )

            if denominator > 1e-12:

                projection = float(
                    np.dot(
                        gps_xy[gps_index]
                        - arc_starts[local_index],
                        vector,
                    ) / denominator
                )

                projection = float(
                    np.clip(projection, 0, 1)
                )

            else:
                projection = 0.0

            bearing_difference = np.nan

            if np.isfinite(gps_bearing):
                bearing_difference = float(
                    angle_difference(
                        arc_bearings[local_index],
                        gps_bearing,
                    )
                )

            records.append({
                "trip_id": trip_id,
                "gps_index": gps_index,
                "timestamp": float(track[gps_index, 2]),
                "arc_id": int(
                    original_arc_ids[local_index]
                ),
                "from_node_id": int(start_node),
                "to_node_id": int(end_node),

                "distance_m": float(distance),
                "bearing_difference_deg": bearing_difference,
                "road_length_m": float(
                    arc_lengths[local_index]
                ),
                "projection_fraction": projection,
                "gps_speed_mps": gps_speeds[gps_index],
                "time_delta_s": time_deltas[gps_index],

                "gps_longitude": float(track[gps_index, 0]),
                "gps_latitude": float(track[gps_index, 1]),
            })

    candidates = pd.DataFrame(records)

    summary = {
        "trip_id": trip_id,
        "gps_observations": len(track),
        "candidate_rows": len(candidates),
        "observations_with_candidates": (
            observations_with_candidates
        ),
        "observations_without_candidates": (
            len(track) - observations_with_candidates
        ),
        "candidate_radius_m": radius_m,
    }

    return candidates, summary


# ---------------------------------------------------------
# SEPARATE EVALUATION-SIDE GROUND TRUTH
# ---------------------------------------------------------

def derive_reference_labels(
    trip_dir: Path,
    trip_id: str,
    track: np.ndarray,
    nodes: np.ndarray,
    arcs: np.ndarray,
):
    """
    Evaluation-only function.

    Loads .route separately. It is never used to generate
    prediction candidates or model features.

    Labels are provisional independent nearest-reference-arc
    assignments, not temporally aligned ground truth.
    """

    route_path = trip_dir / f"{trip_id}.route"

    route = read_table(
        route_path,
        1,
        dtype=np.int64,
    ).ravel()

    if not len(route):
        raise ValueError(
            f"Empty evaluation route for {trip_id}"
        )

    if np.any((route < 0) | (route >= len(arcs))):
        raise ValueError(
            f"Invalid route arc IDs in {trip_id}"
        )

    transformer = create_projection(nodes)

    node_x, node_y = transformer.transform(
        nodes[:, 0],
        nodes[:, 1],
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0],
        track[:, 1],
    )

    node_xy = np.column_stack([
        node_x,
        node_y,
    ])

    gps_xy = np.column_stack([
        gps_x,
        gps_y,
    ])

    route_arcs = arcs[route]

    starts = node_xy[route_arcs[:, 0]]
    ends = node_xy[route_arcs[:, 1]]

    vectors = ends - starts

    lengths_squared = np.maximum(
        np.sum(vectors ** 2, axis=1),
        1e-12,
    )

    best_positions = []
    best_distances = []

    for offset in range(0, len(gps_xy), 128):

        gps = gps_xy[offset:offset + 128]

        difference = (
            gps[:, None, :]
            - starts[None, :, :]
        )

        projection = np.sum(
            difference * vectors[None, :, :],
            axis=2,
        ) / lengths_squared[None, :]

        projection = np.clip(
            projection,
            0,
            1,
        )

        closest = (
            starts[None, :, :]
            + projection[:, :, None]
            * vectors[None, :, :]
        )

        distances = np.linalg.norm(
            gps[:, None, :] - closest,
            axis=2,
        )

        best = np.argmin(
            distances,
            axis=1,
        )

        best_positions.extend(best.tolist())

        best_distances.extend(
            distances[
                np.arange(len(gps)),
                best,
            ].tolist()
        )

    best_positions = np.asarray(
        best_positions,
        dtype=int,
    )

    best_distances = np.asarray(
        best_distances,
        dtype=float,
    )

    reference_arc_ids = route[best_positions]

    return (
        route,
        reference_arc_ids,
        best_positions,
        best_distances,
    )


# ---------------------------------------------------------
# FIT FIVE MODELS ON ALL ACCEPTED TRAINING DATA
# ---------------------------------------------------------

def fit_all_models():

    if not TRAIN_FEATURES.is_file():
        raise FileNotFoundError(TRAIN_FEATURES)

    logger.info("Loading accepted training features...")

    df = pd.read_csv(
        TRAIN_FEATURES,
        dtype={"trip_id": "string"},
        usecols=[
            "trip_id",
            *FEATURES,
            TARGET,
        ],
    )

    if df.empty:
        raise ValueError("Training dataset is empty.")

    X = df[FEATURES].astype(np.float32)

    y = df[TARGET].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "Both classes are required for training."
        )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fitted_models = {}
    training_times = {}

    for name, original_model in make_models().items():

        logger.info("\nFitting %s on all accepted training data", name)

        model = clone(original_model)

        fit_kwargs = {}

        if name == "HistGradientBoosting":

            fit_kwargs[
                "classifier__sample_weight"
            ] = compute_sample_weight(
                class_weight="balanced",
                y=y,
            )

        start = time.perf_counter()

        model.fit(
            X,
            y,
            **fit_kwargs,
        )

        seconds = time.perf_counter() - start

        safe_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.lower(),
        ).strip("_")

        artifact = MODEL_DIR / f"{safe_name}.joblib"

        joblib.dump(
            model,
            artifact,
            compress=3,
        )

        fitted_models[name] = model
        training_times[name] = seconds

        logger.info(
            "%s fitted in %.2f seconds",
            name,
            seconds,
        )

        logger.info("Saved: %s", artifact)

        del model

    metadata = {
        "fit_partition": "train",
        "training_trips": int(df["trip_id"].nunique()),
        "training_rows": len(df),
        "feature_columns": FEATURES,
        "target_column": TARGET,
        "models": training_times,
    }

    with (
        MODEL_DIR / "training_metadata.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4)

    return fitted_models, training_times


# ---------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------

def calculate_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
):

    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    binary_predictions = (
        probabilities >= 0.5
    ).astype(int)

    # Candidate-level metrics.
    candidate_metrics = {
        "candidate_accuracy": accuracy_score(
            labels,
            binary_predictions,
        ),

        "candidate_macro_f1": f1_score(
            labels,
            binary_predictions,
            average="macro",
            zero_division=0,
        ),

        "candidate_precision": precision_score(
            labels,
            binary_predictions,
            zero_division=0,
        ),

        "candidate_recall": recall_score(
            labels,
            binary_predictions,
            zero_division=0,
        ),

        "candidate_log_loss": log_loss(
            labels,
            np.column_stack([
                1 - probabilities,
                probabilities,
            ]),
            labels=[0, 1],
        ),
    }

    # One highest-probability candidate per GPS observation.
    ranked = metadata[
        ["trip_id", "gps_index"]
    ].copy()

    ranked["label"] = labels
    ranked["probability"] = probabilities

    best_indices = (
        ranked.groupby(
            ["trip_id", "gps_index"],
            sort=False,
        )["probability"]
        .idxmax()
    )

    selected = ranked.loc[best_indices]

    observation_accuracy = float(
        selected["label"].mean()
    )

    return {
        **candidate_metrics,
        "observation_top1_accuracy": observation_accuracy,
        "evaluated_observations": len(selected),
    }


# ---------------------------------------------------------
# EVALUATE ONE STUDENT-TEST TRIP
# ---------------------------------------------------------

def evaluate_one_trip(
    trip_dir: Path,
    trip_id: str,
    models: dict,
    radius_m: float,
    label_limit_m: float,
):
    logger.info("\nEvaluating student trip %s", trip_id)

    # Prediction pipeline first.
    # No reference route is read here.
    prediction_data, prediction_summary = (
        generate_inference_candidates(
            trip_dir,
            trip_id,
            radius_m,
        )
    )

    if prediction_data.empty:
        raise ValueError(
            f"No candidate roads generated for {trip_id}"
        )

    X_test = prediction_data[FEATURES].astype(
        np.float32
    )

    predictions_by_model = {}

    for name, model in models.items():

        start = time.perf_counter()

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        inference_seconds = (
            time.perf_counter() - start
        )

        predictions_by_model[name] = probabilities

        prediction_data[f"{name}_probability"] = probabilities

        logger.info(
            "%s inference time: %.4f seconds",
            name,
            inference_seconds,
        )

    # Save predictions without ground-truth labels.
    # This mirrors the intended prediction-side output.
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_files = {}

    for name in models:

        probability_column = f"{name}_probability"

        ranked = prediction_data[
            [
                "trip_id",
                "gps_index",
                "timestamp",
                "arc_id",
                probability_column,
            ]
        ].copy()

        ranked = ranked.rename(
            columns={
                probability_column: "probability"
            }
        )

        best_indices = (
            ranked.groupby(
                ["trip_id", "gps_index"],
                sort=False,
            )["probability"]
            .idxmax()
        )

        best = ranked.loc[best_indices].copy()

        best = best.rename(
            columns={
                "arc_id": "predicted_arc_id",
                "probability": "predicted_probability",
            }
        )

        safe_name = re.sub(
            r"[^a-z0-9]+",
            "_",
            name.lower(),
        ).strip("_")

        prediction_path = (
            OUTPUT_DIR
            / "predictions"
            / f"{trip_id}_{safe_name}.csv"
        )

        prediction_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        best.to_csv(
            prediction_path,
            index=False,
        )

        prediction_files[name] = str(prediction_path)

    # -----------------------------------------------------
    # EVALUATION-SIDE GROUND TRUTH
    # -----------------------------------------------------

    # Route is loaded only after prediction outputs are created.
    track, nodes, arcs = load_trip_without_labels(
        trip_dir,
        trip_id,
    )

    (
        route,
        reference_arc_ids,
        reference_positions,
        reference_distances,
    ) = derive_reference_labels(
        trip_dir,
        trip_id,
        track,
        nodes,
        arcs,
    )

    reference_start = arcs[
        reference_arc_ids, 0
    ]

    reference_end = arcs[
        reference_arc_ids, 1
    ]

    # Match candidates to the selected reference segment.
    ref_map = pd.DataFrame({
        "gps_index": np.arange(len(track)),
        "reference_arc_id": reference_arc_ids,
        "reference_from_node": reference_start,
        "reference_to_node": reference_end,
        "reference_route_position": reference_positions,
        "reference_distance_m": reference_distances,
    })

    prediction_data = prediction_data.merge(
        ref_map,
        on="gps_index",
        how="left",
        validate="many_to_one",
    )

    prediction_data["is_reference"] = (
        (
            prediction_data["from_node_id"]
            == prediction_data["reference_from_node"]
        )
        &
        (
            prediction_data["to_node_id"]
            == prediction_data["reference_to_node"]
        )
    ).astype(int)

    # Observation-level eligibility:
    # reference route near GPS AND represented by a candidate.
    reference_near = (
        ref_map["reference_distance_m"]
        <= label_limit_m
    )

    positive_counts = (
        prediction_data.groupby("gps_index")[
            "is_reference"
        ].sum()
    )

    has_positive = positive_counts.gt(0)

    eligible_indices = set(
        ref_map.loc[
            reference_near,
            "gps_index",
        ]
    ).intersection(
        has_positive[has_positive].index
    )

    total_observations = len(track)

    observations_with_candidates = int(
        prediction_summary["observations_with_candidates"]
    )

    route_near_count = int(reference_near.sum())

    observations_with_positive = int(
        has_positive.sum()
    )

    eligible_count = len(eligible_indices)

    candidate_pool_coverage = (
        observations_with_candidates
        / total_observations * 100
    )

    route_near_percentage = (
        route_near_count
        / total_observations * 100
    )

    valid_reference_coverage = (
        eligible_count
        / total_observations * 100
    )

    evaluation_frame = prediction_data[
        prediction_data["gps_index"].isin(eligible_indices)
    ].copy()

    if not evaluation_frame.empty:

        evaluation_frame = evaluation_frame.sort_values(
            ["gps_index", "arc_id"]
        ).reset_index(drop=True)

    # Prediction outputs already exist independently.
    # Metrics use the separate evaluation reference.
    trip_metrics = []

    for name in models:

        if evaluation_frame.empty:

            metrics = {
                "candidate_accuracy": np.nan,
                "candidate_macro_f1": np.nan,
                "candidate_precision": np.nan,
                "candidate_recall": np.nan,
                "candidate_log_loss": np.nan,
                "observation_top1_accuracy": np.nan,
                "evaluated_observations": 0,
            }

        else:

            metrics = calculate_metrics(
                evaluation_frame["is_reference"].to_numpy(),
                evaluation_frame[f"{name}_probability"].to_numpy(),
                evaluation_frame[
                    ["trip_id", "gps_index"]
                ],
            )

        trip_metrics.append({
            "trip_id": trip_id,
            "model": name,
            "gps_observations": total_observations,
            "observations_with_candidates": (
                observations_with_candidates
            ),
            "candidate_pool_coverage_pct": (
                candidate_pool_coverage
            ),
            "observations_reference_within_threshold": (
                route_near_count
            ),
            "reference_route_near_pct": (
                route_near_percentage
            ),
            "observations_with_reference_candidate": (
                observations_with_positive
            ),
            "eligible_observations": eligible_count,
            "eligible_observation_coverage_pct": (
                valid_reference_coverage
            ),
            "reference_candidate_misses": (
                total_observations - observations_with_positive
            ),
            "reference_distance_threshold_m": label_limit_m,
            "candidate_radius_m": radius_m,
            "reference_alignment_method": (
                "Independent nearest reference-route segment "
                "per GPS observation; threshold-filtered"
            ),
            **metrics,
        })

    return trip_metrics, prediction_summary, prediction_files


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--radius",
        type=float,
        default=CANDIDATE_RADIUS_M,
    )

    parser.add_argument(
        "--label-distance",
        type=float,
        default=LABEL_DISTANCE_LIMIT_M,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of student trips for smoke testing.",
    )

    args = parser.parse_args()

    if args.radius <= 0 or args.label_distance <= 0:
        parser.error("Distances must be positive.")

    config = load_config()

    if not config.get("student_test_dir"):
        raise ValueError(
            "student_test_dir is missing from data.yaml."
        )

    student_dir = resolve_path(
        config["student_test_dir"]
    )

    trip_ids = sorted(
        path.name
        for path in student_dir.iterdir()
        if path.is_dir()
    )

    if args.limit is not None:
        trip_ids = trip_ids[:args.limit]

    logger.info(
        "Student testing trips available for evaluation: %d",
        len(trip_ids),
    )

    # Fit all five models using training data only.
    models, training_times = fit_all_models()

    all_trip_metrics = []
    trip_summaries = []

    for index, trip_id in enumerate(trip_ids, start=1):

        logger.info(
            "\n========== STUDENT TRIP %d/%d ==========",
            index,
            len(trip_ids),
        )

        trip_dir = student_dir / trip_id

        metrics, prediction_summary, prediction_files = (
            evaluate_one_trip(
                trip_dir=trip_dir,
                trip_id=trip_id,
                models=models,
                radius_m=args.radius,
                label_limit_m=args.label_distance,
            )
        )

        all_trip_metrics.extend(metrics)

        trip_summaries.append({
            **prediction_summary,
            "prediction_files": prediction_files,
        })

    per_trip = pd.DataFrame(all_trip_metrics)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_trip.to_csv(
        OUTPUT_DIR / "student_test_metrics_by_trip.csv",
        index=False,
    )

    # -----------------------------------------------------
    # AGGREGATED STUDENT-TEST RESULTS
    # -----------------------------------------------------

    aggregate_rows = []

    metric_columns = [
        "candidate_accuracy",
        "candidate_macro_f1",
        "candidate_precision",
        "candidate_recall",
        "candidate_log_loss",
        "observation_top1_accuracy",
    ]

    for model_name, subset in per_trip.groupby("model"):

        row = {
            "model": model_name,
        }

        # Pool observation-level metrics weighted by
        # evaluated observation count.
        for metric in metric_columns:

            valid = subset[
                subset[metric].notna()
                & (subset["evaluated_observations"] > 0)
            ]

            if valid.empty:
                row[metric] = np.nan
            else:
                row[metric] = float(
                    np.average(
                        valid[metric],
                        weights=valid["evaluated_observations"],
                    )
                )

        row["evaluated_observations"] = int(
            subset["evaluated_observations"].sum()
        )

        row["student_trips"] = int(
            subset["trip_id"].nunique()
        )

        row["mean_trip_top1_accuracy"] = (
            subset["observation_top1_accuracy"].mean()
        )

        row["mean_trip_candidate_macro_f1"] = (
            subset["candidate_macro_f1"].mean()
        )

        row["mean_eligible_observation_coverage_pct"] = (
            subset["eligible_observation_coverage_pct"].mean()
        )

        row["total_training_time_seconds"] = (
            training_times[model_name]
        )

        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)

    if CV_RESULTS.is_file():

        cv = pd.read_csv(CV_RESULTS)

        aggregate = aggregate.merge(
            cv[
                [
                    "model",
                    "mean_cv_macro_f1",
                    "std_cv_macro_f1",
                    "observation_top1_accuracy",
                    "candidate_log_loss",
                    "total_inference_time_seconds",
                ]
            ].rename(
                columns={
                    "observation_top1_accuracy": (
                        "cv_observation_top1_accuracy"
                    ),
                    "candidate_log_loss": "cv_candidate_log_loss",
                }
            ),
            on="model",
            how="left",
            validate="one_to_one",
        )

    aggregate = aggregate.sort_values(
        "observation_top1_accuracy",
        ascending=False,
    ) if "observation_top1_accuracy" in aggregate.columns else aggregate

    aggregate.to_csv(
        OUTPUT_DIR / "student_test_model_comparison.csv",
        index=False,
    )

    manifest = {
        "prediction_partition": "student_test",
        "training_partition": "train",
        "evaluator_testing_accessed": False,
        "models_refitted_on_training_data": list(models.keys()),
        "student_testing_trips": trip_ids,
        "candidate_radius_m": args.radius,
        "label_distance_threshold_m": args.label_distance,
        "prediction_pipeline_reads_route": False,
        "ground_truth_used_only_for_evaluation": True,
        "ground_truth_alignment_method": (
            "Independent nearest reference-route segment per GPS "
            "observation; observations filtered by distance and "
            "candidate availability"
        ),
        "training_times_seconds": training_times,
    }

    with (
        OUTPUT_DIR / "student_test_evaluation_manifest.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    with (
        OUTPUT_DIR / "student_test_generation_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(trip_summaries, file, indent=4)

    logger.info("\n" + "=" * 65)
    logger.info("STUDENT-TESTING EVALUATION COMPLETED")
    logger.info("=" * 65)

    logger.info(
        "\n%s",
        aggregate.to_string(index=False),
    )

    logger.info(
        "\nSaved metrics: %s",
        OUTPUT_DIR / "student_test_model_comparison.csv",
    )

    logger.info(
        "Saved per-trip metrics: %s",
        OUTPUT_DIR / "student_test_metrics_by_trip.csv",
    )


if __name__ == "__main__":
    main()
