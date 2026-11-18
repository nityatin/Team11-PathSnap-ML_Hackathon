
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT.parent / "Student folder"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--trip", default="00000002")
    parser.add_argument("--radius", type=float, default=50)
    args = parser.parse_args()

    trip = DATASET / "train" / args.trip
    prefix = trip / args.trip

    track = np.loadtxt(str(prefix) + ".track")
    nodes = np.loadtxt(str(prefix) + ".nodes")
    arcs = np.loadtxt(str(prefix) + ".arcs", dtype=int)
    route = np.loadtxt(
        str(prefix) + ".route", dtype=int
    ).reshape(-1)

    # Local metric projection
    lon0 = float(np.median(nodes[:, 0]))
    lat0 = float(np.median(nodes[:, 1]))

    crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    transformer = Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    )

    nx_, ny = transformer.transform(
        nodes[:, 0], nodes[:, 1]
    )

    gx, gy = transformer.transform(
        track[:, 0], track[:, 1]
    )

    node_xy = np.column_stack([nx_, ny])
    gps_xy = np.column_stack([gx, gy])

    # Road-network geometries
    starts = node_xy[arcs[:, 0]]
    ends = node_xy[arcs[:, 1]]

    road_geometries = [
        LineString([a, b])
        for a, b in zip(starts, ends)
    ]

    road_tree = STRtree(road_geometries)

    # Reference route geometries
    route_starts = starts[route]
    route_ends = ends[route]

    vectors = route_ends - route_starts

    lengths_sq = np.maximum(
        np.sum(vectors ** 2, axis=1), 1e-12
    )

    route_positions = []
    route_distances = []

    for offset in range(0, len(gps_xy), 128):

        points = gps_xy[offset:offset + 128]

        difference = (
            points[:, None, :] - route_starts[None, :, :]
        )

        projection = np.sum(
            difference * vectors[None, :, :],
            axis=2
        ) / lengths_sq[None, :]

        projection = np.clip(projection, 0, 1)

        closest = (
            route_starts[None, :, :]
            + projection[:, :, None] * vectors[None, :, :]
        )

        distances = np.linalg.norm(
            points[:, None, :] - closest,
            axis=2
        )

        best = np.argmin(distances, axis=1)

        route_positions.extend(best)
        route_distances.extend(
            distances[np.arange(len(points)), best]
        )

    route_positions = np.asarray(route_positions)
    route_distances = np.asarray(route_distances)

    # Nearest segment anywhere in the road network
    network_distances = []

    for x, y in gps_xy:
        point = Point(float(x), float(y))
        nearest_index = int(road_tree.nearest(point))

        network_distances.append(
            road_geometries[nearest_index].distance(point)
        )

    network_distances = np.asarray(network_distances)

    # Report
    print("\n===== TRIP ALIGNMENT DIAGNOSTICS =====")
    print("Trip:", args.trip)
    print("GPS observations:", len(track))
    print("Road arcs:", len(arcs))
    print("Reference route arcs:", len(route))

    print("\nDISTANCE TO REFERENCE ROUTE")

    for p in [50, 75, 90, 95, 99, 100]:
        print(
            f"P{p}: "
            f"{np.percentile(route_distances, p):.2f} m"
        )

    print("\nDISTANCE TO ANY ROAD SEGMENT")

    for p in [50, 75, 90, 95, 99, 100]:
        print(
            f"P{p}: "
            f"{np.percentile(network_distances, p):.2f} m"
        )

    print("\nRADIUS COVERAGE")
    for radius in [25, 50, 75, 100, 150, 250, 500]:

        route_hits = int(
            (route_distances <= radius).sum()
        )

        network_hits = int(
            (network_distances <= radius).sum()
        )

        n = len(track)

        print(
            f"{radius:>3} m | "
            f"Reference route: {route_hits}/{n} "
            f"({route_hits/n*100:.2f}%) | "
            f"Any road: {network_hits}/{n} "
            f"({network_hits/n*100:.2f}%)"
        )

    differences = np.diff(route_positions)

    print("\nROUTE PROGRESSION")
    print("Backward changes:", int((differences < 0).sum()))
    print("First route position:", route_positions[0])
    print("Last route position:", route_positions[-1])

    result = pd.DataFrame({
        "gps_index": np.arange(len(track)),
        "timestamp": track[:, 2],
        "longitude": track[:, 0],
        "latitude": track[:, 1],
        "nearest_route_position": route_positions,
        "nearest_route_arc_id": route[route_positions],
        "distance_to_route_m": route_distances,
        "distance_to_any_road_m": network_distances,
    })

    output_dir = (
        PROJECT_ROOT
        / "reports"
        / "dataset_inspection"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = (
        output_dir
        / f"trip_{args.trip}_alignment_diagnostics.csv"
    )

    result.to_csv(output_file, index=False)

    print("\nFARTHEST 10 GPS OBSERVATIONS FROM REFERENCE ROUTE")

    print(
        result.nlargest(10, "distance_to_route_m")[
            [
                "gps_index",
                "timestamp",
                "distance_to_route_m",
                "distance_to_any_road_m",
                "nearest_route_position",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
