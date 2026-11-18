
import numpy as np
import pandas as pd

from pathlib import Path
from pyproj import CRS, Transformer


ROOT = Path("Student folder/train/00000001/00000001")

track = pd.read_csv(
    str(ROOT) + ".track",
    sep=r"\s+",
    header=None
).to_numpy(dtype=float)

nodes = pd.read_csv(
    str(ROOT) + ".nodes",
    sep=r"\s+",
    header=None
).to_numpy(dtype=float)

arcs = pd.read_csv(
    str(ROOT) + ".arcs",
    sep=r"\s+",
    header=None
).to_numpy(dtype=int)

route = pd.read_csv(
    str(ROOT) + ".route",
    sep=r"\s+",
    header=None
).to_numpy(dtype=int).ravel()


# Local metric coordinate system for this trip
lon0 = float(np.median(nodes[:, 0]))
lat0 = float(np.median(nodes[:, 1]))

local_crs = CRS.from_proj4(
    f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} "
    "+datum=WGS84 +units=m +no_defs"
)

transformer = Transformer.from_crs(
    "EPSG:4326",
    local_crs,
    always_xy=True
)

node_x, node_y = transformer.transform(
    nodes[:, 0],
    nodes[:, 1]
)

track_x, track_y = transformer.transform(
    track[:, 0],
    track[:, 1]
)

node_xy = np.column_stack([node_x, node_y])
track_xy = np.column_stack([track_x, track_y])

# Convert reference arc IDs into their road segments.
route_arcs = arcs[route]

start_points = node_xy[route_arcs[:, 0]]
end_points = node_xy[route_arcs[:, 1]]

vectors = end_points - start_points

length_squared = np.sum(
    vectors * vectors,
    axis=1
)

length_squared = np.maximum(length_squared, 1e-12)

nearest_positions = []
nearest_distances = []

# Calculate GPS distance to every reference-route segment.
for start in range(0, len(track_xy), 128):

    gps = track_xy[start:start + 128]

    difference = gps[:, None, :] - start_points[None, :, :]

    projection = np.sum(
        difference * vectors[None, :, :],
        axis=2
    ) / length_squared[None, :]

    projection = np.clip(projection, 0, 1)

    closest = (
        start_points[None, :, :]
        + projection[:, :, None] * vectors[None, :, :]
    )

    distances = np.linalg.norm(
        gps[:, None, :] - closest,
        axis=2
    )

    best = np.argmin(distances, axis=1)

    nearest_positions.extend(best.tolist())

    nearest_distances.extend(
        distances[np.arange(len(gps)), best].tolist()
    )


nearest_positions = np.asarray(nearest_positions)
nearest_distances = np.asarray(nearest_distances)

print("\n===== GPS TO REFERENCE ROUTE ALIGNMENT =====")

print("GPS observations:", len(track))
print("Reference route arcs:", len(route))

print("\nNearest route distance (metres):")

for percentile in [50, 75, 90, 95, 99, 100]:
    print(
        f"P{percentile}: "
        f"{np.percentile(nearest_distances, percentile):.2f}"
    )

print("\nObservations within route distance:")

for radius in [10, 25, 50, 75, 100]:
    count = int((nearest_distances <= radius).sum())

    print(
        f"{radius} m: {count}/{len(track)} "
        f"({count / len(track) * 100:.2f}%)"
    )

print("\nRoute progression:")

backward = np.sum(np.diff(nearest_positions) < 0)

print("First route position:", nearest_positions[0])
print("Last route position:", nearest_positions[-1])
print("Unique route arcs nearest to GPS:", len(np.unique(nearest_positions)))
print("Backward position changes:", backward)

print("\nFirst 10 GPS observations:")

for i in range(min(10, len(track))):
    position = nearest_positions[i]

    print(
        f"GPS {i}: "
        f"timestamp={track[i, 2]:.1f}, "
        f"route_position={position}, "
        f"arc_id={route[position]}, "
        f"distance={nearest_distances[i]:.2f}m"
    )

print("\nFarthest 10 GPS observations:")

for i in np.argsort(nearest_distances)[-10:][::-1]:
    print(
        f"GPS {i}: "
        f"distance={nearest_distances[i]:.2f}m, "
        f"route_position={nearest_positions[i]}"
    )

# Save alignment diagnostics.
result = pd.DataFrame({
    "gps_index": np.arange(len(track)),
    "timestamp": track[:, 2],
    "nearest_route_position": nearest_positions,
    "nearest_route_arc_id": route[nearest_positions],
    "distance_to_reference_route_m": nearest_distances,
})

output = Path(
    "PathSnap/reports/dataset_inspection"
)

output.mkdir(parents=True, exist_ok=True)

result.to_csv(
    output / "trip_00000001_alignment.csv",
    index=False
)

print("\nAlignment CSV saved successfully.")
