
from __future__ import annotations

import argparse
import json
import logging

from pathlib import Path

import pandas as pd
import yaml

from build_candidates import build_candidate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

CANDIDATE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "candidate_generation"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "training_dataset"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = logging.getLogger("PathSnap.TrainingDataset")


def load_config():

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def get_trip_directories(config):

    train_dir = Path(config["train_dir"])

    if not train_dir.is_absolute():
        train_dir = (PROJECT_ROOT / train_dir).resolve()

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {train_dir}"
        )

    return sorted(
        path.name
        for path in train_dir.iterdir()
        if path.is_dir()
    )


def get_or_generate_candidates(
    trip_id: str,
    radius: float,
    overwrite: bool,
):

    csv_path = (
        CANDIDATE_DIR
        / f"train_{trip_id}_candidates.csv"
    )

    summary_path = (
        CANDIDATE_DIR
        / f"train_{trip_id}_summary.json"
    )

    # Reuse a completed result only if its radius matches.
    if (
        not overwrite
        and csv_path.is_file()
        and summary_path.is_file()
    ):

        with summary_path.open(
            "r", encoding="utf-8"
        ) as file:
            summary = json.load(file)

        if (
            summary.get("candidate_radius_m") == radius
            and csv_path.stat().st_size > 0
        ):
            logger.info(
                "Reusing existing candidates for trip %s",
                trip_id
            )
            return summary, csv_path

    logger.info(
        "Generating candidates for trip %s",
        trip_id
    )

    summary = build_candidate_dataset(
        partition="train",
        trip_id=trip_id,
        radius_m=radius,
    )

    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Candidate output missing for {trip_id}"
        )

    return summary, csv_path


def append_csv(
    source: Path,
    destination,
    first_file: bool,
):

    with source.open("r", encoding="utf-8") as reader:

        header = reader.readline()

        if not header:
            raise ValueError(
                f"Candidate CSV has no header: {source}"
            )

        if first_file:
            destination.write(header)

        for line in reader:
            destination.write(line)


def count_labels(csv_path: Path):

    positive = 0
    negative = 0

    for chunk in pd.read_csv(
        csv_path,
        usecols=["is_reference"],
        chunksize=100_000,
    ):

        positive += int(
            (chunk["is_reference"] == 1).sum()
        )

        negative += int(
            (chunk["is_reference"] == 0).sum()
        )

    return positive, negative


def main():

    parser = argparse.ArgumentParser(
        description="Build candidate dataset from training trips."
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional trip limit for testing."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing per-trip candidate files."
    )

    args = parser.parse_args()

    if args.radius <= 0:
        parser.error("Radius must be greater than zero.")

    if args.limit is not None and args.limit < 1:
        parser.error("Limit must be positive.")

    config = load_config()

    trip_ids = get_trip_directories(config)

    actual_trip_count = len(trip_ids)

    if actual_trip_count != 80:
        logger.warning(
            "Expected 80 training trips, found %d.",
            actual_trip_count
        )

    if args.limit is not None:
        trip_ids = trip_ids[:args.limit]

    # Preview and full results are stored separately.
    is_full_run = (
        args.limit is None
        and len(trip_ids) == actual_trip_count
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if is_full_run:
        output_name = "training_candidates.csv"
    else:
        output_name = "training_candidates_preview.csv"

    final_csv = OUTPUT_DIR / output_name

    temporary_csv = final_csv.with_suffix(".csv.tmp")

    coverage_records = []

    total_observations = 0
    total_candidate_rows = 0
    total_observations_with_candidates = 0
    total_observations_with_reference = 0

    total_positive = 0
    total_negative = 0

    logger.info(
        "Starting training candidate generation: %d trips",
        len(trip_ids)
    )

    # Start a fresh combined file.
    with temporary_csv.open(
        "w",
        encoding="utf-8"
    ) as master:

        for index, trip_id in enumerate(
            trip_ids,
            start=1
        ):

            logger.info(
                "\n========== TRIP %d/%d: %s ==========",
                index,
                len(trip_ids),
                trip_id
            )

            summary, candidate_path = (
                get_or_generate_candidates(
                    trip_id=trip_id,
                    radius=args.radius,
                    overwrite=args.overwrite,
                )
            )

            append_csv(
                candidate_path,
                master,
                first_file=(index == 1),
            )

            observations = int(
                summary["gps_observations"]
            )

            candidates = int(
                summary["candidate_rows"]
            )

            covered = int(
                summary["observations_with_candidates"]
            )

            reference_hits = int(
                summary[
                    "observations_with_reference_candidate"
                ]
            )

            total_observations += observations
            total_candidate_rows += candidates

            total_observations_with_candidates += covered

            total_observations_with_reference += (
                reference_hits
            )

            coverage_records.append({
                "trip_id": trip_id,
                "gps_observations": observations,
                "candidate_rows": candidates,
                "candidate_coverage_pct": (
                    covered / observations * 100
                    if observations else 0
                ),
                "reference_candidate_recall_pct": (
                    reference_hits / observations * 100
                    if observations else 0
                ),
                "candidate_misses": (
                    observations - reference_hits
                ),
                "average_candidates": summary[
                    "average_candidates_per_observation"
                ],
                "maximum_candidates": summary[
                    "maximum_candidates_per_observation"
                ],
                "candidate_radius_m": args.radius,
            })

            logger.info(
                "Completed trip %s: %d candidates, "
                "reference coverage %.2f%%",
                trip_id,
                candidates,
                (
                    reference_hits / observations * 100
                    if observations else 0
                )
            )

            # Save coverage progress after every trip.
            pd.DataFrame(
                coverage_records
            ).to_csv(
                OUTPUT_DIR / "training_candidate_coverage.csv",
                index=False
            )

    # Replace final file only after successful completion.
    temporary_csv.replace(final_csv)

    total_positive, total_negative = count_labels(final_csv)

    summary = {
        "dataset": "PathSnap",
        "partition": "train",
        "evaluator_data_accessed": False,
        "student_test_data_accessed": False,
        "trips_processed": len(trip_ids),
        "expected_training_trips": actual_trip_count,
        "candidate_radius_m": args.radius,
        "gps_observations": total_observations,
        "total_candidate_rows": total_candidate_rows,
        "positive_candidates": total_positive,
        "negative_candidates": total_negative,
        "positive_candidate_percentage": (
            total_positive / total_candidate_rows * 100
            if total_candidate_rows else 0
        ),
        "negative_candidate_percentage": (
            total_negative / total_candidate_rows * 100
            if total_candidate_rows else 0
        ),
        "observations_with_candidates": (
            total_observations_with_candidates
        ),
        "observations_with_reference_candidate": (
            total_observations_with_reference
        ),
        "reference_candidate_recall_pct": (
            total_observations_with_reference
            / total_observations * 100
            if total_observations else 0
        ),
        "observations_with_candidate_misses": (
            total_observations
            - total_observations_with_reference
        ),
        "combined_candidate_csv": str(final_csv),
    }

    summary_path = (
        OUTPUT_DIR / "training_dataset_summary.json"
    )

    with summary_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING DATASET GENERATION COMPLETED")
    logger.info("=" * 60)

    for key, value in summary.items():
        logger.info("%s: %s", key, value)

    logger.info("Saved combined CSV: %s", final_csv)
    logger.info("Saved summary: %s", summary_path)


if __name__ == "__main__":
    main()
