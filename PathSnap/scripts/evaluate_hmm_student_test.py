
from pathlib import Path
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_student_test as ev
from pathsnap.decoder.viterbi import decode_viterbi

RADIUS = 50.0
LABEL_LIMIT = 50.0

MANIFEST = ROOT / "experiments/results/student_testing/student_test_evaluation_manifest.json"
OUTPUT = ROOT / "experiments/results/hmm_evaluation"

MODELS = {
    "Random Forest": ROOT / "artifacts/models/student_testing_evaluation/random_forest.joblib",
    "Bayesian Random Forest": ROOT / "artifacts/models/bayesian_optimized_random_forest.joblib",
}


def reference_labels(trip_dir, trip_id):
    track, nodes, arcs = ev.load_trip_without_labels(trip_dir, trip_id)

    route = np.loadtxt(
        trip_dir / f"{trip_id}.route",
        dtype=np.int64
    ).reshape(-1)

    transformer = ev.create_projection(nodes)

    nx, ny = transformer.transform(nodes[:, 0], nodes[:, 1])
    gx, gy = transformer.transform(track[:, 0], track[:, 1])

    node_xy = np.column_stack([nx, ny])
    gps_xy = np.column_stack([gx, gy])

    route_pairs = arcs[route]
    starts = node_xy[route_pairs[:, 0]]
    ends = node_xy[route_pairs[:, 1]]

    vectors = ends - starts
    lengths = np.maximum(np.sum(vectors ** 2, axis=1), 1e-12)

    labels = []
    distances = []

    for offset in range(0, len(gps_xy), 64):
        gps = gps_xy[offset:offset + 64]

        diff = gps[:, None, :] - starts[None, :, :]
        projection = np.sum(
            diff * vectors[None, :, :], axis=2
        ) / lengths[None, :]

        projection = np.clip(projection, 0, 1)
        closest = starts[None, :, :] + projection[:, :, None] * vectors[None, :, :]

        d = np.linalg.norm(gps[:, None, :] - closest, axis=2)
        best = np.argmin(d, axis=1)

        labels.extend(route[best].tolist())
        distances.extend(d[np.arange(len(gps)), best].tolist())

    return np.asarray(labels), np.asarray(distances)


def decode_contiguous_groups(groups):
    """Decode consecutive GPS observations separately."""
    result = {}

    if not groups:
        return result

    indices = sorted(groups)
    segments = []
    current = []

    for index in indices:
        if current and index != current[-1] + 1:
            segments.append(current)
            current = []
        current.append(index)

    if current:
        segments.append(current)

    for segment in segments:
        decoded = decode_viterbi(
            [groups[index] for index in segment],
            disconnected_penalty=8.0,
        )

        for index, candidate in zip(segment, decoded):
            result[index] = int(candidate["arc_id"])

    return result


def main():
    with MANIFEST.open() as f:
        manifest = json.load(f)

    config = ev.load_config()
    student_dir = ev.resolve_path(config["student_test_dir"])

    trips = manifest["student_testing_trips"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []

    models = {
        name: joblib.load(path)
        for name, path in MODELS.items()
    }

    for trip_id in trips:
        print(f"\nProcessing trip {trip_id}")

        trip_dir = student_dir / trip_id

        candidates, _ = ev.generate_inference_candidates(
            trip_dir, trip_id, RADIUS
        )

        if candidates.empty:
            print("No candidates; skipping.")
            continue

        truth, distances = reference_labels(trip_dir, trip_id)

        eligible = distances <= LABEL_LIMIT

        for model_name, model in models.items():
            start = time.perf_counter()

            probabilities = model.predict_proba(
                candidates[ev.FEATURES].astype(np.float32)
            )[:, 1]

            inference_seconds = time.perf_counter() - start

            work = candidates[
                ["gps_index", "arc_id", "from_node_id", "to_node_id"]
            ].copy()

            work["probability"] = probabilities

            groups = {}
            baseline = {}

            for gps_index, group in work.groupby("gps_index", sort=True):
                gps_index = int(gps_index)

                group = group.sort_values(
                    "probability", ascending=False
                )

                baseline[gps_index] = int(group.iloc[0]["arc_id"])

                groups[gps_index] = [
                    {
                        "arc_id": int(row.arc_id),
                        "from_node_id": int(row.from_node_id),
                        "to_node_id": int(row.to_node_id),
                        "probability": float(row.probability),
                    }
                    for row in group.itertuples()
                ]

            hmm_predictions = decode_contiguous_groups(groups)

            common = [
                i for i in baseline
                if i < len(truth) and eligible[i]
            ]

            if not common:
                continue

            baseline_accuracy = np.mean([
                baseline[i] == truth[i] for i in common
            ])

            hmm_accuracy = np.mean([
                hmm_predictions[i] == truth[i] for i in common
            ])

            results.append({
                "trip_id": trip_id,
                "model": model_name,
                "evaluated_observations": len(common),
                "baseline_top1_accuracy": baseline_accuracy,
                "hmm_top1_accuracy": hmm_accuracy,
                "improvement_percentage_points":
                    (hmm_accuracy - baseline_accuracy) * 100,
                "inference_seconds": inference_seconds,
            })

            print(
                f"{model_name}: "
                f"RF={baseline_accuracy:.4f}, "
                f"HMM={hmm_accuracy:.4f}"
            )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT / "hmm_comparison_by_trip.csv", index=False)

    summary = (
        df.groupby("model")[
            [
                "baseline_top1_accuracy",
                "hmm_top1_accuracy",
                "improvement_percentage_points",
            ]
        ].mean()
    )

    summary.to_csv(OUTPUT / "hmm_comparison_summary.csv")

    print("\n===== FINAL HMM COMPARISON =====")
    print(summary.to_string())
    print("\nSaved results to:", OUTPUT)


if __name__ == "__main__":
    main()
