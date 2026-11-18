
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import yaml

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import evaluate_student_test as ev
from pathsnap.decoder.viterbi import decode_viterbi

# Configuration
with (ROOT / "configs" / "data.yaml").open() as f:
    CONFIG = yaml.safe_load(f)

EVALUATOR_DIR = ev.resolve_path(CONFIG["evaluator_test_dir"])

MODEL_PATH = (
    ROOT / "artifacts" / "models"
    / "bayesian_optimized_random_forest.joblib"
)

RADIUS_M = 50.0

# Load model once at startup
model = joblib.load(MODEL_PATH)

app = FastAPI(title="PathSnap API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://10.154.247.31:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_trip(trip_id):
    trip_dir = EVALUATOR_DIR / trip_id

    if not trip_dir.is_dir():
        raise HTTPException(404, "Evaluator test trip not found")

    prefix = trip_dir / trip_id

    track = ev.read_table(
        Path(str(prefix) + ".track"), 3
    )
    nodes = ev.read_table(
        Path(str(prefix) + ".nodes"), 2
    )
    arcs = ev.read_table(
        Path(str(prefix) + ".arcs"), 2, dtype=np.int64
    )

    return trip_dir, track, nodes, arcs


def decode_groups(groups):
    """Decode contiguous GPS-index ranges independently."""
    predictions = {}
    indices = sorted(groups)

    chunks = []
    current = []

    for index in indices:
        if current and index != current[-1] + 1:
            chunks.append(current)
            current = []
        current.append(index)

    if current:
        chunks.append(current)

    for chunk in chunks:
        decoded = decode_viterbi(
            [groups[i] for i in chunk],
            disconnected_penalty=8.0,
        )

        for index, candidate in zip(chunk, decoded):
            predictions[index] = candidate

    return predictions


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Bayesian Random Forest + HMM/Viterbi",
    }


@app.get("/api/v1/trips")
def list_trips():
    trips = sorted(
        p.name for p in EVALUATOR_DIR.iterdir()
        if p.is_dir()
    )
    return {"trips": trips, "count": len(trips)}


@app.post("/api/v1/map-match/{trip_id}")
def map_match(trip_id: str, use_hmm: bool = True):
    start_time = time.perf_counter()

    trip_dir, track, nodes, arcs = read_trip(trip_id)

    # Prediction-side candidate generation.
    # This function reads .track, .nodes and .arcs, never .route.
    candidates, _ = ev.generate_inference_candidates(
        trip_dir, trip_id, RADIUS_M
    )

    if candidates.empty:
        raise HTTPException(422, "No road candidates generated")

    X = candidates[ev.FEATURES].astype(np.float32)
    probabilities = model.predict_proba(X)[:, 1]

    candidates = candidates.copy()
    candidates["probability"] = probabilities

    groups = {}

    for gps_index, group in candidates.groupby(
        "gps_index", sort=True
    ):
        groups[int(gps_index)] = [
            {
                "arc_id": int(row.arc_id),
                "from_node_id": int(row.from_node_id),
                "to_node_id": int(row.to_node_id),
                "probability": float(row.probability),
            }
            for row in group.itertuples()
        ]

    # Classifier-only predictions.
    baseline = {
        index: max(
            group, key=lambda x: x["probability"]
        )
        for index, group in groups.items()
    }

    # Optional sequence decoding.
    selected = (
        decode_groups(groups) if use_hmm else baseline
    )

    segments = []
    geometry = []

    for gps_index in sorted(selected):
        candidate = selected[gps_index]

        start_node = candidate["from_node_id"]
        end_node = candidate["to_node_id"]

        if (
            start_node >= len(nodes)
            or end_node >= len(nodes)
        ):
            continue

        start_coord = [
            float(nodes[start_node, 0]),
            float(nodes[start_node, 1]),
        ]
        end_coord = [
            float(nodes[end_node, 0]),
            float(nodes[end_node, 1]),
        ]

        geometry.append([start_coord, end_coord])

        segments.append({
            "gps_index": gps_index,
            "arc_id": candidate["arc_id"],
            "probability": candidate["probability"],
        })

    elapsed = time.perf_counter() - start_time

    return {
        "trip_id": trip_id,
        "status": "success",
        "model": "Bayesian Random Forest",
        "decoder": "HMM + Viterbi" if use_hmm else "None",
        "inference_time_seconds": round(elapsed, 4),
        "gps_points": [
            {
                "longitude": float(row[0]),
                "latitude": float(row[1]),
                "timestamp": float(row[2]),
            }
            for row in track
        ],
        "predicted_segments": segments,
        "route_geometry": {
            "type": "MultiLineString",
            "coordinates": geometry,
        },
        "metrics": {
            "top1_accuracy": None,
            "note": "Ground-truth labels are not used during prediction.",
        },
    }
