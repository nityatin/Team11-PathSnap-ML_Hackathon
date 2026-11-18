
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "dataset_inspection"

EXPECTED_FILES = {
    "track": (".track", 3, False),
    "nodes": (".nodes", 2, False),
    "arcs": (".arcs", 2, True),
    "route": (".route", 1, True),
}

EARTH_RADIUS_KM = 6371.0088

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger("PathSnap.DatasetInspection")


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Invalid data.yaml configuration.")

    return config


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ---------------------------------------------------------
# RAW DATA READER
# ---------------------------------------------------------

def read_numeric_table(
    path: Path,
    expected_columns: int,
    integer: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Reads a whitespace-separated numeric file.

    Important:
    Malformed rows are retained as NaN placeholders.
    This preserves original zero-based row indices,
    which define node IDs and arc IDs in this dataset.
    """

    rows = []
    malformed_rows = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="replace"
    ) as file:

        for line in file:
            parts = line.split()

            if not parts:
                continue

            if len(parts) != expected_columns:
                rows.append([np.nan] * expected_columns)
                malformed_rows += 1
                continue

            values = []
            row_invalid = False

            for token in parts:
                try:
                    value = float(token)

                    if not np.isfinite(value):
                        value = np.nan
                        row_invalid = True

                    elif integer and not np.isclose(
                        value,
                        round(value),
                        rtol=0,
                        atol=1e-9
                    ):
                        value = np.nan
                        row_invalid = True

                except ValueError:
                    value = np.nan
                    row_invalid = True

                values.append(value)

            if row_invalid:
                malformed_rows += 1

            rows.append(values)

    if rows:
        array = np.asarray(rows, dtype=np.float64)
    else:
        array = np.empty(
            (0, expected_columns),
            dtype=np.float64
        )

    valid_rows = np.isfinite(array).all(axis=1)

    return array, valid_rows, malformed_rows


# ---------------------------------------------------------
# GEOGRAPHICAL HELPERS
# ---------------------------------------------------------

def coordinate_validity(
    coordinates: np.ndarray
) -> np.ndarray:
    """
    Coordinates are stored as longitude, latitude.
    """

    return (
        np.isfinite(coordinates).all(axis=1)
        & (coordinates[:, 0] >= -180)
        & (coordinates[:, 0] <= 180)
        & (coordinates[:, 1] >= -90)
        & (coordinates[:, 1] <= 90)
    )


def haversine_km(
    lon1: np.ndarray,
    lat1: np.ndarray,
    lon2: np.ndarray,
    lat2: np.ndarray,
) -> np.ndarray:

    lon1 = np.radians(lon1)
    lat1 = np.radians(lat1)
    lon2 = np.radians(lon2)
    lat2 = np.radians(lat2)

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    value = np.clip(value, 0, 1)

    return (
        2
        * EARTH_RADIUS_KM
        * np.arcsin(np.sqrt(value))
    )


def coordinate_extent(
    coordinates: np.ndarray,
    valid_mask: np.ndarray
) -> dict[str, float | None]:

    valid = coordinates[valid_mask]

    if len(valid) == 0:
        return {
            "longitude_min": None,
            "longitude_max": None,
            "latitude_min": None,
            "latitude_max": None,
        }

    return {
        "longitude_min": float(valid[:, 0].min()),
        "longitude_max": float(valid[:, 0].max()),
        "latitude_min": float(valid[:, 1].min()),
        "latitude_max": float(valid[:, 1].max()),
    }


# ---------------------------------------------------------
# TRIP INSPECTION
# ---------------------------------------------------------

def inspect_trip(
    partition: str,
    trip_dir: Path,
) -> tuple[dict, dict, dict, dict, list]:

    trip_id = trip_dir.name

    tables = {}
    row_masks = {}
    file_records = []

    trip_size = 0
    missing_files = 0

    # Read all four expected files.
    for name, (extension, columns, is_integer) in EXPECTED_FILES.items():

        file_path = trip_dir / f"{trip_id}{extension}"

        exists = file_path.is_file()

        if exists:
            size = file_path.stat().st_size
            trip_size += size

            array, valid_mask, malformed = read_numeric_table(
                file_path,
                columns,
                integer=is_integer,
            )

            tables[name] = array
            row_masks[name] = valid_mask

            raw_rows = len(array)
            valid_count = int(valid_mask.sum())

        else:
            missing_files += 1
            size = 0
            malformed = 0
            raw_rows = 0
            valid_count = 0

            tables[name] = np.empty((0, columns))
            row_masks[name] = np.zeros(0, dtype=bool)

        file_records.append({
            "partition": partition,
            "trip_id": trip_id,
            "file_type": name,
            "file_name": file_path.name,
            "exists": exists,
            "size_bytes": size,
            "raw_rows": raw_rows,
            "valid_rows": valid_count,
            "malformed_rows": malformed,
        })

    # -----------------------------------------------------
    # GPS TRACK INSPECTION
    # -----------------------------------------------------

    track = tables["track"]

    if len(track):
        coordinates = track[:, :2]
        gps_valid = coordinate_validity(coordinates)

        timestamps = track[:, 2]
        timestamp_valid = np.isfinite(timestamps)

        valid_timestamps = timestamps[timestamp_valid]
        time_deltas = np.diff(valid_timestamps)

        exact_one_second = np.isclose(
            time_deltas, 1.0, rtol=0, atol=1e-6
        )

        nonpositive_gaps = int((time_deltas <= 0).sum())
        gaps_over_one_second = int((time_deltas > 1).sum())

        exact_one_second_pct = (
            float(exact_one_second.mean() * 100)
            if len(time_deltas)
            else None
        )

        # Calculate distance only between consecutive
        # valid coordinate observations.
        if len(track) > 1:
            consecutive_valid = gps_valid[:-1] & gps_valid[1:]

            distances = haversine_km(
                coordinates[:-1, 0][consecutive_valid],
                coordinates[:-1, 1][consecutive_valid],
                coordinates[1:, 0][consecutive_valid],
                coordinates[1:, 1][consecutive_valid],
            )

            trajectory_length_km = float(distances.sum())

            valid_distance_pairs = int(consecutive_valid.sum())

        else:
            trajectory_length_km = 0.0
            valid_distance_pairs = 0

        extent = coordinate_extent(coordinates, gps_valid)

        timestamp_min = (
            float(valid_timestamps.min())
            if len(valid_timestamps) else None
        )

        timestamp_max = (
            float(valid_timestamps.max())
            if len(valid_timestamps) else None
        )

        nonpositive_timestamp_count = nonpositive_gaps

    else:
        gps_valid = np.zeros(0, dtype=bool)
        trajectory_length_km = 0.0
        valid_distance_pairs = 0
        exact_one_second_pct = None
        gaps_over_one_second = 0
        nonpositive_timestamp_count = 0

        extent = coordinate_extent(
            np.empty((0, 2)),
            np.zeros(0, dtype=bool)
        )

        timestamp_min = None
        timestamp_max = None
        valid_timestamps = np.empty(0)

    trajectory_record = {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observation_rows": len(track),
        "valid_coordinate_rows": int(gps_valid.sum()),
        "invalid_coordinate_rows": int((~gps_valid).sum()),
        "valid_timestamp_rows": len(valid_timestamps),
        "timestamp_min": timestamp_min,
        "timestamp_max": timestamp_max,
        "timestamp_delta_count": max(len(valid_timestamps) - 1, 0),
        "nonpositive_timestamp_deltas": nonpositive_timestamp_count,
        "gaps_over_one_second": gaps_over_one_second,
        "exact_one_second_percentage": exact_one_second_pct,
        "valid_consecutive_distance_pairs": valid_distance_pairs,
        "trajectory_length_km": trajectory_length_km,
        **extent,
    }

    # -----------------------------------------------------
    # ROAD NETWORK INSPECTION
    # -----------------------------------------------------

    nodes = tables["nodes"]
    arcs = tables["arcs"]

    node_count = len(nodes)
    arc_count = len(arcs)

    if node_count:
        node_coordinates = nodes[:, :2]
        node_valid = coordinate_validity(node_coordinates)
    else:
        node_valid = np.zeros(0, dtype=bool)

    arc_parse_valid = row_masks["arcs"]

    # Convert only valid integer rows.
    arc_integer = np.zeros(
        (arc_count, 2),
        dtype=np.int64
    )

    if arc_parse_valid.any():
        arc_integer[arc_parse_valid] = np.rint(
            arcs[arc_parse_valid]
        ).astype(np.int64)

    if arc_count:
        endpoint_valid = (
            (arc_integer[:, 0] >= 0)
            & (arc_integer[:, 0] < node_count)
            & (arc_integer[:, 1] >= 0)
            & (arc_integer[:, 1] < node_count)
        )

        valid_arc_mask = arc_parse_valid & endpoint_valid

    else:
        valid_arc_mask = np.zeros(0, dtype=bool)

    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))

    if valid_arc_mask.any():
        graph.add_edges_from(
            map(tuple, arc_integer[valid_arc_mask])
        )

    if node_count:
        components = list(nx.connected_components(graph))

        component_count = len(components)

        largest_component_nodes = max(
            (len(component) for component in components),
            default=0
        )

        isolated_nodes = sum(
            1 for _, degree in graph.degree()
            if degree == 0
        )

    else:
        component_count = 0
        largest_component_nodes = 0
        isolated_nodes = 0

    valid_arc_pairs = [
        tuple(row) for row in arc_integer[valid_arc_mask]
    ]

    duplicate_arc_rows = (
        len(valid_arc_pairs) - len(set(valid_arc_pairs))
    )

    self_loop_count = sum(
        1 for start, end in valid_arc_pairs
        if start == end
    )

    node_extent = coordinate_extent(
        nodes[:, :2] if node_count else np.empty((0, 2)),
        node_valid
    )

    network_record = {
        "partition": partition,
        "trip_id": trip_id,
        "node_rows": node_count,
        "valid_node_coordinates": int(node_valid.sum()),
        "invalid_node_coordinates": int((~node_valid).sum()),
        "arc_rows": arc_count,
        "valid_arcs": int(valid_arc_mask.sum()),
        "invalid_arcs": int((~valid_arc_mask).sum()),
        "isolated_nodes": isolated_nodes,
        "connected_components": component_count,
        "largest_component_nodes": largest_component_nodes,
        "duplicate_arc_rows": duplicate_arc_rows,
        "self_loop_count": self_loop_count,
        **node_extent,
    }

    # -----------------------------------------------------
    # REFERENCE ROUTE INSPECTION
    # -----------------------------------------------------

    route = tables["route"]
    route_parse_valid = row_masks["route"]

    route_ids = np.zeros(len(route), dtype=np.int64)

    if route_parse_valid.any():
        route_ids[route_parse_valid] = np.rint(
            route[route_parse_valid, 0]
        ).astype(np.int64)

    route_in_bounds = (
        route_parse_valid
        & (route_ids >= 0)
        & (route_ids < arc_count)
    )

    route_references_invalid_arcs = np.zeros(
        len(route), dtype=bool
    )

    if route_in_bounds.any():
        positions = np.flatnonzero(route_in_bounds)

        route_references_invalid_arcs[positions] = (
            ~valid_arc_mask[route_ids[positions]]
        )

    valid_route_mask = (
        route_in_bounds
        & ~route_references_invalid_arcs
    )

    # Test whether consecutive route arcs connect:
    # end_node(previous_arc) == start_node(next_arc)
    pair_valid = (
        valid_route_mask[:-1]
        & valid_route_mask[1:]
    )

    previous_ids = route_ids[:-1][pair_valid]
    next_ids = route_ids[1:][pair_valid]

    continuity_checks = len(previous_ids)

    if continuity_checks:
        continuous_pairs = (
            arc_integer[previous_ids, 1]
            == arc_integer[next_ids, 0]
        )

        disconnected_transitions = int(
            (~continuous_pairs).sum()
        )

        route_continuity_pct = float(
            continuous_pairs.mean() * 100
        )

    else:
        disconnected_transitions = 0
        route_continuity_pct = None

    route_record = {
        "partition": partition,
        "trip_id": trip_id,
        "route_rows": len(route),
        "valid_route_arc_ids": int(valid_route_mask.sum()),
        "malformed_route_rows": int((~route_parse_valid).sum()),
        "out_of_range_arc_ids": int(
            (route_parse_valid & ~route_in_bounds).sum()
        ),
        "references_to_invalid_arcs": int(
            route_references_invalid_arcs.sum()
        ),
        "continuity_checks": continuity_checks,
        "disconnected_transitions": disconnected_transitions,
        "route_continuity_percentage": route_continuity_pct,
    }

    # -----------------------------------------------------
    # FILE INVENTORY AND QUALITY
    # -----------------------------------------------------

    malformed_total = sum(
        record["malformed_rows"] for record in file_records
    )

    trip_record = {
        "partition": partition,
        "trip_id": trip_id,
        "has_track": (trip_dir / f"{trip_id}.track").is_file(),
        "has_nodes": (trip_dir / f"{trip_id}.nodes").is_file(),
        "has_arcs": (trip_dir / f"{trip_id}.arcs").is_file(),
        "has_route": (trip_dir / f"{trip_id}.route").is_file(),
        "complete_trip": missing_files == 0,
        "missing_files": missing_files,
        "total_size_bytes": trip_size,
        "track_rows": len(track),
        "node_rows": node_count,
        "arc_rows": arc_count,
        "route_rows": len(route),
    }

    quality_record = {
        "partition": partition,
        "trip_id": trip_id,
        "missing_files": missing_files,
        "malformed_rows_all_files": malformed_total,
        "invalid_gps_coordinates": int((~gps_valid).sum()),
        "nonpositive_timestamp_deltas": nonpositive_timestamp_count,
        "invalid_node_coordinates": int((~node_valid).sum()),
        "invalid_arcs": int((~valid_arc_mask).sum()),
        "invalid_route_arc_references": int(
            (~valid_route_mask).sum()
        ),
        "disconnected_route_transitions": disconnected_transitions,
    }

    return (
        trip_record,
        trajectory_record,
        network_record,
        route_record,
        file_records,
    ), quality_record


# ---------------------------------------------------------
# MAIN DATASET INSPECTION
# ---------------------------------------------------------

def main() -> None:

    config = load_config()

    # Evaluator-testing data is intentionally excluded.
    partitions = [
        ("train", config.get("train_dir")),
        ("student_test", config.get("student_test_dir")),
    ]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    trip_inventory = []
    trajectory_stats = []
    network_stats = []
    route_stats = []
    file_inventory = []
    quality_records = []

    partition_summary = {}

    for partition, configured_path in partitions:

        if not configured_path:
            raise ValueError(
                f"Missing path configuration for {partition}."
            )

        partition_dir = resolve_path(configured_path)

        if not partition_dir.is_dir():
            raise FileNotFoundError(
                f"{partition} directory not found: {partition_dir}"
            )

        trip_dirs = sorted(
            path for path in partition_dir.iterdir()
            if path.is_dir()
        )

        logger.info(
            "Inspecting %s: %d trip directories",
            partition,
            len(trip_dirs)
        )

        partition_start = len(trip_inventory)

        for index, trip_dir in enumerate(trip_dirs, start=1):

            logger.info(
                "[%s] %d/%d - Trip %s",
                partition,
                index,
                len(trip_dirs),
                trip_dir.name
            )

            result, quality = inspect_trip(
                partition,
                trip_dir
            )

            (
                trip_record,
                trajectory_record,
                network_record,
                route_record,
                file_records,
            ) = result

            trip_inventory.append(trip_record)
            trajectory_stats.append(trajectory_record)
            network_stats.append(network_record)
            route_stats.append(route_record)
            file_inventory.extend(file_records)
            quality_records.append(quality)

        current_trips = trip_inventory[partition_start:]

        partition_summary[partition] = {
            "trip_directories": len(current_trips),
            "complete_trips": sum(
                row["complete_trip"] for row in current_trips
            ),
            "missing_files": sum(
                row["missing_files"] for row in current_trips
            ),
            "total_size_bytes": sum(
                row["total_size_bytes"] for row in current_trips
            ),
        }

    # -----------------------------------------------------
    # SAVE REPORTS
    # -----------------------------------------------------

    outputs = {
        "trip_inventory.csv": trip_inventory,
        "file_inventory.csv": file_inventory,
        "trajectory_stats.csv": trajectory_stats,
        "network_stats.csv": network_stats,
        "route_stats.csv": route_stats,
        "data_quality.csv": quality_records,
    }

    for filename, records in outputs.items():

        dataframe = pd.DataFrame(records)

        dataframe.to_csv(
            OUTPUT_DIR / filename,
            index=False
        )

    summary = {
        "dataset": "PathSnap",
        "partitions_inspected": [
            "train",
            "student_test"
        ],
        "evaluator_testing_inspected": False,
        "partition_summary": partition_summary,
        "total_trips_inspected": len(trip_inventory),
        "total_files_found": sum(
            record["exists"] for record in file_inventory
        ),
        "total_files_expected": len(file_inventory),
        "complete_trips": sum(
            record["complete_trip"] for record in trip_inventory
        ),
        "total_malformed_rows": sum(
            record["malformed_rows_all_files"]
            for record in quality_records
        ),
        "output_directory": str(OUTPUT_DIR),
    }

    with (OUTPUT_DIR / "dataset_summary.json").open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=4)

    # -----------------------------------------------------
    # CONSOLE SUMMARY
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print(" PATHSNAP - DATASET INSPECTION COMPLETED")
    print("=" * 60)

    print(f"Total trips inspected : {len(trip_inventory)}")
    print(f"Complete trips        : {summary['complete_trips']}")
    print(f"Files found           : {summary['total_files_found']}")
    print(f"Files expected        : {summary['total_files_expected']}")
    print(f"Malformed rows        : {summary['total_malformed_rows']}")

    for name, details in partition_summary.items():
        print(f"\n{name.upper()}")
        print(f"  Trips        : {details['trip_directories']}")
        print(f"  Complete     : {details['complete_trips']}")
        print(f"  Missing files: {details['missing_files']}")

    print("\nGenerated reports:")

    for filename in outputs:
        print(f"  - {OUTPUT_DIR / filename}")

    print(f"  - {OUTPUT_DIR / 'dataset_summary.json'}")

    print("\nEvaluator-testing data was not inspected.")
    print("=" * 60)


if __name__ == "__main__":
    main()
