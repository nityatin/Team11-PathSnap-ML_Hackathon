
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import networkx as nx

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

EARTH_RADIUS_KM = 6371.0088


def load_config():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: Path, columns: int, dtype=float):
    data = np.loadtxt(path, dtype=dtype)

    data = np.asarray(data)

    if data.size == 0:
        return np.empty((0, columns), dtype=dtype)

    return data.reshape(-1, columns)


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
        always_xy=True
    )


def calculate_reference_alignment(
    gps_xy: np.ndarray,
    nodes_xy: np.ndarray,
    arcs: np.ndarray,
    route: np.ndarray,
    chunk_size: int = 64,
):
    """
    Finds the nearest reference-route arc for every GPS point.

    This is an initial independent nearest-segment alignment.
    It does not enforce monotonic route progression.
    """

    route_arcs = arcs[route]

    starts = nodes_xy[route_arcs[:, 0]]
    ends = nodes_xy[route_arcs[:, 1]]

    vectors = ends - starts

    lengths_squared = np.maximum(
        np.sum(vectors ** 2, axis=1),
        1e-12
    )

    positions = []
    distances = []

    for offset in range(0, len(gps_xy), chunk_size):

        gps = gps_xy[offset:offset + chunk_size]

        difference = (
            gps[:, None, :] - starts[None, :, :]
        )

        projection = np.sum(
            difference * vectors[None, :, :],
            axis=2
        ) / lengths_squared[None, :]

        projection = np.clip(projection, 0, 1)

        closest = (
            starts[None, :, :]
            + projection[:, :, None]
            * vectors[None, :, :]
        )

        distance = np.linalg.norm(
            gps[:, None, :] - closest,
            axis=2
        )

        best = np.argmin(distance, axis=1)

        positions.extend(best.tolist())

        distances.extend(
            distance[np.arange(len(gps)), best].tolist()
        )

    positions = np.asarray(positions, dtype=int)
    distances = np.asarray(distances, dtype=float)

    return positions, route[positions], distances


def calculate_gps_motion(
    gps_xy: np.ndarray,
    timestamps: np.ndarray,
):
    """
    Calculates local movement bearing and speed
    from consecutive GPS observations.
    """

    count = len(gps_xy)

    bearings = np.full(count, np.nan)
    speeds = np.full(count, np.nan)
    time_deltas = np.full(count, np.nan)

    for i in range(count):

        if count < 2:
            break

        if i < count - 1:
            start = i
            end = i + 1
        else:
            start = i - 1
            end = i

        displacement = gps_xy[end] - gps_xy[start]

        dx, dy = displacement

        distance = float(np.hypot(dx, dy))

        dt = float(timestamps[end] - timestamps[start])

        time_deltas[i] = dt

        if distance > 1e-6:
            bearings[i] = (
                np.degrees(np.arctan2(dx, dy)) + 360
            ) % 360

        if dt > 0:
            speeds[i] = distance / dt

    return bearings, speeds, time_deltas


def angle_difference(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def build_candidate_dataset(
    partition: str,
    trip_id: str,
    radius_m: float,
):
    config = load_config()

    if partition != "train":
        raise ValueError(
            "Candidate training dataset generation is restricted "
            "to the training partition."
        )

    train_dir = Path(config["train_dir"])

    if not train_dir.is_absolute():
        train_dir = (PROJECT_ROOT / train_dir).resolve()

    trip_dir = train_dir / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(
            f"Trip directory not found: {trip_dir}"
        )

    print(f"Loading trip {trip_id}...")

    prefix = trip_dir / trip_id

    track = read_table(
        Path(str(prefix) + ".track"), 3
    )

    nodes = read_table(
        Path(str(prefix) + ".nodes"), 2
    )

    arcs = read_table(
        Path(str(prefix) + ".arcs"), 2, dtype=np.int64
    )

    route = read_table(
        Path(str(prefix) + ".route"), 1, dtype=np.int64
    ).ravel()

    if not len(track) or not len(nodes) or not len(arcs):
        raise ValueError("Trip contains empty required data.")

    if not len(route):
        raise ValueError("Reference route is empty.")

    # -----------------------------------------------------
    # COORDINATE PROJECTION
    # -----------------------------------------------------

    transformer = create_projection(nodes)

    node_x, node_y = transformer.transform(
        nodes[:, 0],
        nodes[:, 1]
    )

    gps_x, gps_y = transformer.transform(
        track[:, 0],
        track[:, 1]
    )

    nodes_xy = np.column_stack([node_x, node_y])
    gps_xy = np.column_stack([gps_x, gps_y])

    # -----------------------------------------------------
    # VALIDATE ROAD NETWORK
    # -----------------------------------------------------

    valid_arcs = (
        (arcs[:, 0] >= 0)
        & (arcs[:, 0] < len(nodes))
        & (arcs[:, 1] >= 0)
        & (arcs[:, 1] < len(nodes))
    )

    original_arc_ids = np.flatnonzero(valid_arcs)
    valid_arc_pairs = arcs[valid_arcs]

    if not len(original_arc_ids):
        raise ValueError("No valid road arcs found.")

    arc_starts = nodes_xy[valid_arc_pairs[:, 0]]
    arc_ends = nodes_xy[valid_arc_pairs[:, 1]]

    arc_vectors = arc_ends - arc_starts

    arc_lengths = np.linalg.norm(
        arc_vectors,
        axis=1
    )

    arc_bearings = (
        np.degrees(
            np.arctan2(
                arc_vectors[:, 0],
                arc_vectors[:, 1]
            )
        ) + 360
    ) % 360

    geometries = [
        LineString([start, end])
        for start, end in zip(arc_starts, arc_ends)
    ]

    print("Building road-network spatial index...")

    tree = STRtree(geometries)

    # -----------------------------------------------------
    # REFERENCE ROUTE ALIGNMENT
    # -----------------------------------------------------

    if np.any((route < 0) | (route >= len(arcs))):
        raise ValueError("Reference route contains invalid arc IDs.")

    if not np.all(valid_arcs[route]):
        raise ValueError(
            "Reference route points to an invalid road arc."
        )

    route_positions, reference_arc_ids, route_distances = (
        calculate_reference_alignment(
            gps_xy,
            nodes_xy,
            arcs,
            route,
        )
    )

    # -----------------------------------------------------
    # GPS MOTION FEATURES
    # -----------------------------------------------------

    gps_bearings, gps_speeds, time_deltas = (
        calculate_gps_motion(
            gps_xy,
            track[:, 2],
        )
    )

    # -----------------------------------------------------
    # CANDIDATE GENERATION
    # -----------------------------------------------------

    records = []

    candidate_counts = np.zeros(len(track), dtype=int)
    candidate_hits = np.zeros(len(track), dtype=bool)

    for gps_index, (x, y) in enumerate(gps_xy):

        point = Point(float(x), float(y))

        # Query a square bounding region first.
        # Exact circular distance is checked afterwards.
        search_box = box(
            x - radius_m,
            y - radius_m,
            x + radius_m,
            y + radius_m
        )

        local_indices = np.asarray(
            tree.query(search_box),
            dtype=int
        )

        if not len(local_indices):
            continue

        local_distances = np.asarray([
            geometries[index].distance(point)
            for index in local_indices
        ])

        within_radius = local_distances <= radius_m

        local_indices = local_indices[within_radius]
        local_distances = local_distances[within_radius]

        if not len(local_indices):
            continue

        candidate_counts[gps_index] = len(local_indices)

        ref_arc = int(reference_arc_ids[gps_index])
        ref_position = int(route_positions[gps_index])

        ref_start, ref_end = arcs[ref_arc]

        gps_bearing = gps_bearings[gps_index]

        for local_index, distance in zip(
            local_indices,
            local_distances
        ):

            arc_id = int(original_arc_ids[local_index])

            start_node, end_node = arcs[arc_id]

            # Exact directed edge correspondence.
            # Identical duplicate directed edges are also positive.
            is_reference = (
                start_node == ref_start
                and end_node == ref_end
            )

            if is_reference:
                candidate_hits[gps_index] = True

            bearing_difference = np.nan

            if np.isfinite(gps_bearing):
                bearing_difference = float(
                    angle_difference(
                        arc_bearings[local_index],
                        gps_bearing,
                    )
                )

            vector = arc_vectors[local_index]

            denominator = float(
                np.dot(vector, vector)
            )

            if denominator > 1e-12:

                projection = float(
                    np.dot(
                        gps_xy[gps_index]
                        - arc_starts[local_index],
                        vector
                    ) / denominator
                )

                projection = float(
                    np.clip(projection, 0, 1)
                )

            else:
                projection = 0.0

            records.append({
                "trip_id": trip_id,
                "partition": partition,
                "gps_index": gps_index,
                "timestamp": float(track[gps_index, 2]),

                # Candidate identity
                "arc_id": arc_id,
                "from_node_id": int(start_node),
                "to_node_id": int(end_node),

                # Target derived from training reference route
                "reference_arc_id": ref_arc,
                "reference_route_position": ref_position,
                "is_reference": int(is_reference),

                # Candidate-ranking features
                "distance_m": float(distance),
                "gps_bearing_deg": gps_bearing,
                "road_bearing_deg": float(
                    arc_bearings[local_index]
                ),
                "bearing_difference_deg": bearing_difference,
                "road_length_m": float(
                    arc_lengths[local_index]
                ),
                "projection_fraction": projection,
                "gps_speed_mps": gps_speeds[gps_index],
                "time_delta_s": time_deltas[gps_index],

                # Alignment diagnostics
                "reference_route_distance_m": float(
                    route_distances[gps_index]
                ),

                # Coordinates retained for mapping,
                # not intended as initial ML features.
                "gps_longitude": float(track[gps_index, 0]),
                "gps_latitude": float(track[gps_index, 1]),
            })

    # -----------------------------------------------------
    # SAVE OUTPUT
    # -----------------------------------------------------

    dataframe = pd.DataFrame(records)

    output_dir = (
        PROJECT_ROOT
        / "experiments"
        / "results"
        / "candidate_generation"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        output_dir
        / f"{partition}_{trip_id}_candidates.csv"
    )

    dataframe.to_csv(csv_path, index=False)

    observations_with_candidates = int(
        (candidate_counts > 0).sum()
    )

    observations_with_reference = int(
        candidate_hits.sum()
    )

    candidate_misses = int(
        len(track) - observations_with_reference
    )

    summary = {
        "trip_id": trip_id,
        "partition": partition,
        "gps_observations": len(track),
        "valid_road_arcs": int(len(original_arc_ids)),
        "reference_route_arcs": len(route),
        "candidate_radius_m": radius_m,
        "candidate_rows": len(dataframe),
        "observations_with_candidates": observations_with_candidates,
        "observations_with_reference_candidate": observations_with_reference,
        "candidate_misses": candidate_misses,
        "candidate_miss_rate_percent": (
            candidate_misses / len(track) * 100
        ),
        "average_candidates_per_observation": float(
            candidate_counts.mean()
        ),
        "median_candidates_per_observation": float(
            np.median(candidate_counts)
        ),
        "maximum_candidates_per_observation": int(
            candidate_counts.max()
        ),
        "reference_distance_p95_m": float(
            np.percentile(route_distances, 95)
        ),
        "alignment_backward_changes": int(
            np.sum(np.diff(route_positions) < 0)
        ),
        "candidate_csv": str(csv_path),
    }

    summary_path = (
        output_dir
        / f"{partition}_{trip_id}_summary.json"
    )

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 55)
    print(" PATHSNAP - CANDIDATE GENERATION")
    print("=" * 55)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved candidate dataset:", csv_path)
    print("Saved summary:", summary_path)

    # A binary-classification dataset needs both classes.
    if len(dataframe):
        print("\nCandidate label distribution:")
        print(
            dataframe["is_reference"]
            .value_counts()
            .rename(index={0: "Negative", 1: "Positive"})
            .to_string()
        )

    if candidate_misses:
        print(
            "\nWARNING: Some GPS observations do not have "
            "their reference edge in the candidate set."
        )

    return summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trip",
        default="00000001",
        help="Training trip ID"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=50.0,
        help="Candidate search radius in metres"
    )

    args = parser.parse_args()

    if args.radius <= 0:
        parser.error("Radius must be positive.")

    build_candidate_dataset(
        partition="train",
        trip_id=args.trip,
        radius_m=args.radius,
    )


if __name__ == "__main__":
    main()
