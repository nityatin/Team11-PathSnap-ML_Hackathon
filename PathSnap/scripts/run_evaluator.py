from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_student_test as ev
from pathsnap.decoder.viterbi import decode_viterbi

RADIUS = 50.0
MODEL_PATH = ROOT / "artifacts/models/bayesian_optimized_random_forest.joblib"
OUTPUT = ROOT / "reports/maps/evaluator_test"

def decode_groups(groups):
    result = {}
    indices = sorted(groups)
    chunks, current = [], []

    for i in indices:
        if current and i != current[-1] + 1:
            chunks.append(current)
            current = []
        current.append(i)

    if current:
        chunks.append(current)

    for chunk in chunks:
        decoded = decode_viterbi(
            [groups[i] for i in chunk],
            disconnected_penalty=8.0,
        )
        for i, candidate in zip(chunk, decoded):
            result[i] = int(candidate["arc_id"])

    return result

def main():
    config = ev.load_config()
    evaluator_dir = ev.resolve_path(config["evaluator_test_dir"])

    model = joblib.load(MODEL_PATH)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_rows = []

    trip_dirs = sorted(p for p in evaluator_dir.iterdir() if p.is_dir())
    print(f"Found {len(trip_dirs)} evaluator trips")

    for trip_dir in trip_dirs:
        trip_id = trip_dir.name
        print(f"\nProcessing {trip_id}...")
        start = time.perf_counter()

        # This loader reads ONLY .track, .nodes and .arcs.
        track, nodes, arcs = ev.load_trip_without_labels(trip_dir, trip_id)

        candidates, _ = ev.generate_inference_candidates(
            trip_dir, trip_id, RADIUS
        )

        if candidates.empty:
            print("No candidates; skipping.")
            continue

        probabilities = model.predict_proba(
            candidates[ev.FEATURES].astype(np.float32)
        )[:, 1]

        work = candidates[
            ["gps_index", "arc_id", "from_node_id", "to_node_id"]
        ].copy()
        work["probability"] = probabilities

        groups = {}
        baseline = {}
        baseline_probability = {}

        for gps_index, group in work.groupby("gps_index", sort=True):
            gps_index = int(gps_index)
            group = group.sort_values("probability", ascending=False)

            best = group.iloc[0]
            baseline[gps_index] = int(best.arc_id)
            baseline_probability[gps_index] = float(best.probability)

            groups[gps_index] = [
                {
                    "arc_id": int(row.arc_id),
                    "from_node_id": int(row.from_node_id),
                    "to_node_id": int(row.to_node_id),
                    "probability": float(row.probability),
                }
                for row in group.itertuples()
            ]

        hmm = decode_groups(groups)

        transformer = ev.create_projection(nodes)
        gps_lon = track[:, 0]
        gps_lat = track[:, 1]

        rows = []
        for i in sorted(baseline):
            row = {
                "trip_id": trip_id,
                "gps_index": i,
                "timestamp": track[i, 2],
                "classifier_arc_id": baseline[i],
                "hmm_arc_id": hmm[i],
                "classifier_probability": baseline_probability[i],
            }
            rows.append(row)
            all_rows.append(row)

        pd.DataFrame(rows).to_csv(
            OUTPUT / f"trip_{trip_id}_predictions.csv", index=False
        )

        # Create Folium map.
        center = [float(np.mean(gps_lat)), float(np.mean(gps_lon))]
        m = folium.Map(location=center, zoom_start=14)

        folium.PolyLine(
            list(zip(gps_lat, gps_lon)),
            color="blue",
            weight=3,
            opacity=0.7,
            tooltip="GPS trajectory",
        ).add_to(m)

        gps_layer = folium.FeatureGroup(name="GPS observations")
        for idx in range(0, len(track), max(1, len(track) // 150)):
            folium.CircleMarker(
                location=[gps_lat[idx], gps_lon[idx]],
                radius=2,
                color="blue",
                fill=True,
                tooltip=f"GPS {idx}",
            ).add_to(gps_layer)
        gps_layer.add_to(m)

        predicted_layer = folium.FeatureGroup(
            name="Bayesian RF + HMM predicted roads"
        )

        for arc_id in sorted(set(hmm.values())):
            if arc_id < 0 or arc_id >= len(arcs):
                continue

            a, b = arcs[arc_id]
            if a >= len(nodes) or b >= len(nodes):
                continue

            folium.PolyLine(
                [[nodes[a, 1], nodes[a, 0]],
                 [nodes[b, 1], nodes[b, 0]]],
                color="red",
                weight=5,
                opacity=0.85,
                tooltip=f"Predicted road arc {arc_id}",
            ).add_to(predicted_layer)

        predicted_layer.add_to(m)

        folium.Marker(
            [gps_lat[0], gps_lon[0]],
            tooltip="Start",
            icon=folium.Icon(color="green"),
        ).add_to(m)

        folium.Marker(
            [gps_lat[-1], gps_lon[-1]],
            tooltip="End",
            icon=folium.Icon(color="red"),
        ).add_to(m)

        folium.LayerControl().add_to(m)

        map_path = OUTPUT / f"trip_{trip_id}_evaluator_map.html"
        m.save(str(map_path))

        print(
            f"Saved {len(rows)} predictions | "
            f"map: {map_path.name} | "
            f"time: {time.perf_counter() - start:.2f}s"
        )

    if all_rows:
        pd.DataFrame(all_rows).to_csv(
            OUTPUT / "evaluator_all_predictions.csv", index=False
        )

    print("\nEVALUATOR PREDICTION COMPLETE")
    print("Output folder:", OUTPUT)
    print("No evaluator route labels were loaded or evaluated.")

if __name__ == "__main__":
    main()
