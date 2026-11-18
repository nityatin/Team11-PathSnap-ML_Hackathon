
from __future__ import annotations

import argparse
import json
from pathlib import Path

import folium
import numpy as np
import yaml

from pyproj import CRS, Transformer
from shapely.geometry import LineString, Point, box
from shapely.strtree import STRtree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

OUTPUT_DIR = PROJECT_ROOT / "reports" / "maps" / "alignment"

# Geographical map padding in projected metres.
MAP_PADDING_M = 350

# Maximum roads rendered in the background layer.
MAX_ROAD_FEATURES = 40000

# Maximum GPS markers, in addition to the full trajectory line.
MAX_GPS_MARKERS = 500


def read_table(path: Path, columns: int, dtype=float):
    data = np.loadtxt(path, dtype=dtype)
    return np.asarray(data).reshape(-1, columns)


def load_trip(trip_id: str):
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    train_dir = Path(config["train_dir"])

    if not train_dir.is_absolute():
        train_dir = (PROJECT_ROOT / train_dir).resolve()

    trip_dir = train_dir / trip_id

    if not trip_dir.is_dir():
        raise FileNotFoundError(trip_dir)

    prefix = trip_dir / trip_id

    track = read_table(
        Path(str(prefix) + ".track"), 3
    )

    nodes = read_table(
        Path(str(prefix) + ".nodes"), 2
    )

    arcs = read_table(
        Path(str(prefix) + ".arcs"), 2, dtype=int
    )

    route = read_table(
        Path(str(prefix) + ".route"), 1, dtype=int
    ).ravel()

    return track, nodes, arcs, route


def geojson_feature(coords, properties=None):
    return {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": coords,
        },
        "properties": properties or {},
    }


def add_geojson_layer(
    fmap,
    features,
    name,
    color,
    weight,
    opacity,
    dash=None,
):
    if not features:
        return

    group = folium.FeatureGroup(
        name=name,
        show=True
    )

    style = {
        "color": color,
        "weight": weight,
        "opacity": opacity,
    }

    if dash:
        style["dashArray"] = dash

    folium.GeoJson(
        {
            "type": "FeatureCollection",
            "features": features,
        },
        style_function=lambda feature: style,
        tooltip=folium.GeoJsonTooltip(
            fields=["description"],
            aliases=["Details:"],
        ),
    ).add_to(group)

    group.add_to(fmap)


def make_map(trip_id: str):

    print(f"\nLoading trip {trip_id}...")

    track, nodes, arcs, route = load_trip(trip_id)

    # Validate references.
    if (
        np.any(route < 0)
        or np.any(route >= len(arcs))
    ):
        raise ValueError("Invalid route arc IDs.")

    if (
        np.any(arcs < 0)
        or np.any(arcs >= len(nodes))
    ):
        raise ValueError("Invalid node references.")

    # -------------------------------------------------
    # LOCAL METRIC PROJECTION
    # -------------------------------------------------

    lon0 = float(np.median(nodes[:, 0]))
    lat0 = float(np.median(nodes[:, 1]))

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
        "+datum=WGS84 +units=m +no_defs"
    )

    forward = Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True,
    )

    inverse = Transformer.from_crs(
        local_crs,
        "EPSG:4326",
        always_xy=True,
    )

    node_x, node_y = forward.transform(
        nodes[:, 0],
        nodes[:, 1],
    )

    gps_x, gps_y = forward.transform(
        track[:, 0],
        track[:, 1],
    )

    node_xy = np.column_stack([node_x, node_y])
    gps_xy = np.column_stack([gps_x, gps_y])

    # -------------------------------------------------
    # ROAD-NETWORK GEOMETRIES
    # -------------------------------------------------

    starts = node_xy[arcs[:, 0]]
    ends = node_xy[arcs[:, 1]]

    geometries = [
        LineString([start, end])
        for start, end in zip(starts, ends)
    ]

    tree = STRtree(geometries)

    # -------------------------------------------------
    # REFERENCE ROUTE
    # -------------------------------------------------

    route_arcs = arcs[route]

    route_starts = nodes[route_arcs[:, 0]]
    route_ends = nodes[route_arcs[:, 1]]

    route_coords = []

    for start, end in zip(route_starts, route_ends):

        route_coords.append([
            [float(start[0]), float(start[1])],
            [float(end[0]), float(end[1])],
        ])

    route_features = []

    for position, (arc_id, coords) in enumerate(
        zip(route, route_coords)
    ):

        route_features.append(
            geojson_feature(
                coords,
                {
                    "description": (
                        f"Reference route position {position}, "
                        f"arc ID {int(arc_id)}"
                    )
                }
            )
        )

    # -------------------------------------------------
    # CROP ROAD NETWORK AROUND GPS + REFERENCE ROUTE
    # -------------------------------------------------

    route_xy = nodes[route_arcs].reshape(-1, 2)

    route_x, route_y = forward.transform(
        route_xy[:, 0],
        route_xy[:, 1],
    )

    combined_x = np.concatenate([gps_x, route_x])
    combined_y = np.concatenate([gps_y, route_y])

    extent = box(
        combined_x.min() - MAP_PADDING_M,
        combined_y.min() - MAP_PADDING_M,
        combined_x.max() + MAP_PADDING_M,
        combined_y.max() + MAP_PADDING_M,
    )

    selected = np.asarray(
        tree.query(extent),
        dtype=int
    )

    # Avoid drawing duplicate/reverse edges twice.
    road_features = []
    seen_edges = set()

    if len(selected) > MAX_ROAD_FEATURES:
        print(
            f"Large road layer ({len(selected)} edges). "
            f"Sampling {MAX_ROAD_FEATURES} for display."
        )

        rng = np.random.default_rng(42)

        selected = np.sort(
            rng.choice(
                selected,
                MAX_ROAD_FEATURES,
                replace=False
            )
        )

    for arc_id in selected:

        start_id, end_id = arcs[arc_id]

        key = tuple(sorted((int(start_id), int(end_id))))

        if key in seen_edges:
            continue

        seen_edges.add(key)

        start = nodes[int(start_id)]
        end = nodes[int(end_id)]

        road_features.append(
            geojson_feature(
                [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ],
                {
                    "description": f"Road arc {int(arc_id)}"
                }
            )
        )

    # -------------------------------------------------
    # NEAREST ROAD TO EACH GPS OBSERVATION
    # -------------------------------------------------

    nearest_roads = {}

    for index, (x, y) in enumerate(gps_xy):

        point = Point(float(x), float(y))

        nearest_arc = int(tree.nearest(point))

        distance = float(
            geometries[nearest_arc].distance(point)
        )

        start_id, end_id = arcs[nearest_arc]

        # Deduplicate reverse-direction edges for display.
        key = tuple(sorted((int(start_id), int(end_id))))

        if (
            key not in nearest_roads
            or distance < nearest_roads[key]["distance"]
        ):
            nearest_roads[key] = {
                "arc_id": nearest_arc,
                "distance": distance,
                "gps_index": index,
                "start": int(start_id),
                "end": int(end_id),
            }

    nearest_features = []

    for record in nearest_roads.values():

        start = nodes[record["start"]]
        end = nodes[record["end"]]

        nearest_features.append(
            geojson_feature(
                [
                    [float(start[0]), float(start[1])],
                    [float(end[0]), float(end[1])],
                ],
                {
                    "description": (
                        f"Nearest road arc {record['arc_id']}; "
                        f"GPS distance {record['distance']:.1f} m"
                    )
                }
            )
        )

    # -------------------------------------------------
    # MAP CREATION
    # -------------------------------------------------

    fmap = folium.Map(
        location=[
            float(np.median(track[:, 1])),
            float(np.median(track[:, 0])),
        ],
        zoom_start=14,
        tiles="CartoDB positron",
        control_scale=True,
    )

    # Background road geometry.
    add_geojson_layer(
        fmap,
        road_features,
        "Road network (cropped geographical extent)",
        "#9299A1",
        1,
        0.45,
    )

    # Nearest roads.
    add_geojson_layer(
        fmap,
        nearest_features,
        "Nearest road to each GPS observation",
        "#A855F7",
        3,
        0.8,
    )

    # Supplied reference route.
    add_geojson_layer(
        fmap,
        route_features,
        "Supplied reference route",
        "#16A34A",
        5,
        0.95,
    )

    # Original GPS trajectory.
    gps_group = folium.FeatureGroup(
        name="Original GPS trajectory",
        show=True,
    )

    gps_latlon = [
        [float(lat), float(lon)]
        for lon, lat in track[:, :2]
    ]

    folium.PolyLine(
        gps_latlon,
        color="#2563EB",
        weight=3,
        opacity=0.8,
        tooltip="Original GPS trajectory",
    ).add_to(gps_group)

    # Sample GPS points to avoid thousands of markers.
    step = max(
        1,
        int(np.ceil(len(track) / MAX_GPS_MARKERS))
    )

    for index in range(0, len(track), step):

        lon, lat, timestamp = track[index]

        folium.CircleMarker(
            location=[float(lat), float(lon)],
            radius=2,
            color="#1D4ED8",
            fill=True,
            fill_opacity=0.75,
            tooltip=(
                f"GPS index: {index}, "
                f"timestamp: {timestamp:.1f}"
            ),
        ).add_to(gps_group)

    gps_group.add_to(fmap)

    # Start and end markers.
    folium.Marker(
        gps_latlon[0],
        tooltip="GPS start",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(fmap)

    folium.Marker(
        gps_latlon[-1],
        tooltip="GPS end",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(fmap)

    # Layer selector.
    folium.LayerControl(
        collapsed=False
    ).add_to(fmap)

    # Add a descriptive title.
    title = f"""
    <div style="
        position: fixed;
        top: 10px;
        left: 50px;
        z-index: 9999;
        background: white;
        padding: 9px 14px;
        border-radius: 5px;
        border: 1px solid #777;
        font-size: 14px;
    ">
        <b>PathSnap - Trip {trip_id}</b><br>
        GPS observations: {len(track):,}<br>
        Reference arcs: {len(route):,}<br>
        Roads displayed: {len(road_features):,}
    </div>
    """

    fmap.get_root().html.add_child(
        folium.Element(title)
    )

    fmap.fit_bounds([
        [
            float(min(track[:, 1].min(), route_xy[:, 1].min())),
            float(min(track[:, 0].min(), route_xy[:, 0].min())),
        ],
        [
            float(max(track[:, 1].max(), route_xy[:, 1].max())),
            float(max(track[:, 0].max(), route_xy[:, 0].max())),
        ],
    ])

    # -------------------------------------------------
    # SAVE MAP AND MANIFEST
    # -------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        OUTPUT_DIR / f"trip_{trip_id}_alignment.html"
    )

    fmap.save(str(output_file))

    distances = [
        item["distance"]
        for item in nearest_roads.values()
    ]

    summary = {
        "trip_id": trip_id,
        "gps_observations": len(track),
        "reference_route_arcs": len(route),
        "road_network_arcs_total": len(arcs),
        "road_features_displayed": len(road_features),
        "nearest_road_features": len(nearest_features),
        "gps_markers_displayed": len(range(0, len(track), step)),
        "nearest_road_distance_median": float(
            np.median(distances)
        ),
        "nearest_road_distance_p95": float(
            np.percentile(distances, 95)
        ),
        "map_file": str(output_file),
    }

    summary_file = (
        OUTPUT_DIR / f"trip_{trip_id}_map_summary.json"
    )

    summary_file.write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    print("\nMAP GENERATED")
    print(json.dumps(summary, indent=4))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--trips",
        nargs="+",
        default=["00000033", "00000054"],
    )

    args = parser.parse_args()

    for trip_id in args.trips:
        make_map(trip_id)


if __name__ == "__main__":
    main()
