
from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"

CANDIDATE_FILE = (
    RESULTS_DIR
    / "training_dataset"
    / "training_candidates.csv"
)

COVERAGE_FILE = (
    RESULTS_DIR
    / "training_dataset"
    / "training_candidate_coverage.csv"
)

OUTPUT_DIR = (
    RESULTS_DIR / "training_features"
)

GROUP_COLUMNS = ["trip_id", "gps_index"]

FEATURE_COLUMNS = [
    "distance_m",
    "bearing_difference_deg",
    "road_length_m",
    "projection_fraction",
    "gps_speed_mps",
    "time_delta_s",
]

REQUIRED_COLUMNS = [
    "trip_id",
    "partition",
    "gps_index",
    "arc_id",
    "reference_arc_id",
    "reference_route_position",
    "reference_route_distance_m",
    "is_reference",
    *FEATURE_COLUMNS,
]

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger("PathSnap.FeatureEngineering")


def load_candidate_data() -> pd.DataFrame:

    if not CANDIDATE_FILE.is_file():
        raise FileNotFoundError(CANDIDATE_FILE)

    logger.info("Loading candidate dataset...")

    df = pd.read_csv(
        CANDIDATE_FILE,
        dtype={"trip_id": "string"},
        usecols=lambda column: column in REQUIRED_COLUMNS,
    )

    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"Required candidate columns missing: {sorted(missing)}"
        )

    if not df["partition"].eq("train").all():
        raise ValueError(
            "The training dataset contains a non-training partition."
        )

    return df


def validate_candidate_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:

    logger.info("Validating candidate records...")

    for column in FEATURE_COLUMNS + [
        "reference_route_distance_m",
        "is_reference",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if not df["is_reference"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    # Remove duplicate candidate rows for the same
    # GPS observation and road arc.
    duplicate_count = int(
        df.duplicated(
            subset=["trip_id", "gps_index", "arc_id"]
        ).sum()
    )

    if duplicate_count:
        logger.warning(
            "Removing %d duplicated candidate rows.",
            duplicate_count,
        )

        df = df.drop_duplicates(
            subset=["trip_id", "gps_index", "arc_id"]
        ).copy()

    # Candidate geometry features must be valid.
    mandatory_numeric = [
        "distance_m",
        "road_length_m",
        "projection_fraction",
        "reference_route_distance_m",
    ]

    for column in mandatory_numeric:
        if not np.isfinite(df[column]).all():
            raise ValueError(
                f"Non-finite values found in {column}."
            )

    if (df["distance_m"] < 0).any():
        raise ValueError("Negative candidate distances found.")

    if (df["road_length_m"] < 0).any():
        raise ValueError("Negative road lengths found.")

    if not df["projection_fraction"].between(0, 1).all():
        raise ValueError("Projection fraction outside [0, 1].")

    return df


def find_valid_observations(
    df: pd.DataFrame,
    maximum_reference_distance_m: float,
):

    logger.info("Checking observation-level labels...")

    # Each GPS observation must have a consistent
    # reference assignment across all its candidates.
    observation_stats = (
        df.groupby(
            GROUP_COLUMNS,
            sort=False,
        )
        .agg(
            candidate_count=("arc_id", "size"),

            positive_candidate_count=(
                "is_reference",
                "sum",
            ),

            reference_distance_m=(
                "reference_route_distance_m",
                "first",
            ),

            reference_arc_count=(
                "reference_arc_id",
                "nunique",
            ),

            reference_position_count=(
                "reference_route_position",
                "nunique",
            ),
        )
    )

    inconsistent = (
        (observation_stats["reference_arc_count"] != 1)
        | (observation_stats["reference_position_count"] != 1)
    )

    if inconsistent.any():
        raise ValueError(
            "Inconsistent reference metadata across "
            f"{int(inconsistent.sum())} GPS observations."
        )

    has_positive = (
        observation_stats["positive_candidate_count"] > 0
    )

    close_enough = (
        observation_stats["reference_distance_m"]
        <= maximum_reference_distance_m
    )

    accepted = has_positive & close_enough

    observation_stats["has_positive"] = has_positive
    observation_stats["within_reference_threshold"] = close_enough
    observation_stats["accepted"] = accepted

    accepted_keys = observation_stats.index[
        observation_stats["accepted"]
    ]

    row_keys = pd.MultiIndex.from_frame(
        df[GROUP_COLUMNS]
    )

    filtered = df.loc[
        row_keys.isin(accepted_keys)
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No valid observations remain after filtering."
        )

    # Verify each accepted observation has a positive label.
    accepted_group_labels = (
        filtered.groupby(GROUP_COLUMNS)["is_reference"]
        .sum()
    )

    if (accepted_group_labels <= 0).any():
        raise AssertionError(
            "An accepted observation has no positive candidate."
        )

    return filtered, observation_stats


def create_quality_report(
    observation_stats: pd.DataFrame,
    coverage: pd.DataFrame,
    maximum_reference_distance_m: float,
):

    coverage = coverage.copy()
    coverage["trip_id"] = coverage["trip_id"].astype("string")

    grouped = observation_stats.reset_index()

    rows = []

    for _, trip in coverage.iterrows():

        trip_id = trip["trip_id"]

        subset = grouped[
            grouped["trip_id"] == trip_id
        ]

        available_observations = int(
            trip["gps_observations"]
        )

        candidate_observations = len(subset)

        accepted_observations = int(
            subset["accepted"].sum()
        )

        no_positive = int(
            (~subset["has_positive"]).sum()
        )

        too_far = int(
            (~subset["within_reference_threshold"]).sum()
        )

        no_candidates = (
            available_observations
            - candidate_observations
        )

        rows.append({
            "trip_id": trip_id,
            "gps_observations": available_observations,
            "observations_with_candidates": candidate_observations,
            "observations_without_candidates": no_candidates,
            "observations_without_positive": no_positive,
            "observations_beyond_reference_threshold": too_far,
            "accepted_observations": accepted_observations,
            "accepted_observation_percentage": (
                accepted_observations / available_observations * 100
                if available_observations else 0
            ),
            "candidate_rows_before_filtering": int(
                subset["candidate_count"].sum()
            ),
            "reference_distance_threshold_m": (
                maximum_reference_distance_m
            ),
        })

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser(
        description="Build clean candidate-ranking features."
    )

    parser.add_argument(
        "--max-reference-distance",
        type=float,
        default=50.0,
    )

    args = parser.parse_args()

    if args.max_reference_distance <= 0:
        parser.error("Distance threshold must be positive.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_candidate_data()

    df = validate_candidate_rows(df)

    # Group-level observation filtering.
    filtered, observation_stats = find_valid_observations(
        df,
        args.max_reference_distance,
    )

    # Keep only explicit model inputs and essential metadata.
    output_columns = [
        "trip_id",
        "gps_index",
        "arc_id",
        "from_node_id",
        "to_node_id",
        "is_reference",
        *FEATURE_COLUMNS,
    ]

    available_output_columns = [
        column for column in output_columns
        if column in filtered.columns
    ]

    final_df = filtered[available_output_columns].copy()

    # Handle infinity as missing. Imputation will be
    # learned separately within each training fold.
    final_df[FEATURE_COLUMNS] = (
        final_df[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Remove records with no usable feature values at all.
    all_missing = final_df[FEATURE_COLUMNS].isna().all(axis=1)

    if all_missing.any():
        logger.warning(
            "Dropping %d rows with all features missing.",
            int(all_missing.sum()),
        )

        final_df = final_df.loc[~all_missing].copy()

    # Re-check positive representation after feature filtering.
    group_labels = (
        final_df.groupby(GROUP_COLUMNS)["is_reference"]
        .sum()
    )

    invalid_groups = group_labels[group_labels == 0].index

    if len(invalid_groups):
        invalid_keys = pd.MultiIndex.from_frame(
            final_df[GROUP_COLUMNS]
        )

        final_df = final_df.loc[
            ~invalid_keys.isin(invalid_groups)
        ].copy()

    if final_df.empty:
        raise ValueError("Final feature dataset is empty.")

    # -----------------------------------------------------
    # SAVE DATASET
    # -----------------------------------------------------

    feature_file = OUTPUT_DIR / "training_features.csv"

    final_df.to_csv(
        feature_file,
        index=False,
    )

    # Per-trip filter report.
    coverage = pd.read_csv(
        COVERAGE_FILE,
        dtype={"trip_id": "string"},
    )

    quality = create_quality_report(
        observation_stats,
        coverage,
        args.max_reference_distance,
    )

    quality.to_csv(
        OUTPUT_DIR / "feature_quality_by_trip.csv",
        index=False,
    )

    # Feature schema.
    schema = {
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "is_reference",
        "group_columns": GROUP_COLUMNS,
        "excluded_reference_columns": [
            "reference_arc_id",
            "reference_route_position",
            "reference_route_distance_m",
        ],
        "label_quality_rule": {
            "minimum_requirement": (
                "At least one positive candidate per GPS observation"
            ),
            "maximum_reference_distance_m": (
                args.max_reference_distance
            ),
        },
        "missing_values": (
            "Handle through fold-specific imputation "
            "during model training."
        ),
    }

    with (
        OUTPUT_DIR / "feature_schema.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(schema, file, indent=4)

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    positive = int(
        (final_df["is_reference"] == 1).sum()
    )

    negative = int(
        (final_df["is_reference"] == 0).sum()
    )

    summary = {
        "partition": "train",
        "trips_in_candidate_source": int(
            coverage["trip_id"].nunique()
        ),
        "gps_observations_with_candidates": int(
            len(observation_stats)
        ),
        "gps_observations_accepted": int(
            observation_stats["accepted"].sum()
        ),
        "maximum_reference_distance_m": (
            args.max_reference_distance
        ),
        "candidate_rows_retained": len(final_df),
        "positive_candidates": positive,
        "negative_candidates": negative,
        "positive_percentage": (
            positive / len(final_df) * 100
        ),
        "negative_percentage": (
            negative / len(final_df) * 100
        ),
        "trip_count_retained": int(
            final_df["trip_id"].nunique()
        ),
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "is_reference",
        "candidate_dataset": str(feature_file),
    }

    with (
        OUTPUT_DIR / "feature_dataset_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    logger.info("\nFEATURE ENGINEERING COMPLETED")

    for key, value in summary.items():
        logger.info("%s: %s", key, value)

    logger.info(
        "Feature dataset saved: %s",
        feature_file,
    )

    logger.info(
        "Per-trip quality report: %s",
        OUTPUT_DIR / "feature_quality_by_trip.csv",
    )


if __name__ == "__main__":
    main()
