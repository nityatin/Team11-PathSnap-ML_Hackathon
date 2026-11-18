
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRIPS_DIR = Path("../ML-Hackathon/Student folder/train").resolve()
OUTPUT = ROOT / "experiments/results/candidate_radius_experiments"

RADII = [50, 75, 100]

OUTPUT.mkdir(parents=True, exist_ok=True)

trip_ids = sorted(
    p.name for p in TRIPS_DIR.iterdir()
    if p.is_dir()
)

results = []

for radius in RADII:
    print(f"\n{'='*55}")
    print(f"RUNNING CANDIDATE RADIUS: {radius}m")
    print(f"{'='*55}")

    summaries = []

    for i, trip_id in enumerate(trip_ids, 1):
        print(f"[{i}/{len(trip_ids)}] Trip {trip_id}")

        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/build_candidates.py"),
                "--trip", trip_id,
                "--radius", str(radius),
            ],
            check=True,
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
        )

        summary_path = (
            ROOT / "experiments/results/candidate_generation"
            / f"train_{trip_id}_summary.json"
        )

        with summary_path.open() as f:
            summaries.append(json.load(f))

    total_observations = sum(
        x["gps_observations"] for x in summaries
    )

    total_hits = sum(
        x["observations_with_reference_candidate"]
        for x in summaries
    )

    total_candidates = sum(
        x["candidate_rows"] for x in summaries
    )

    results.append({
        "radius_m": radius,
        "candidate_recall_pct": 100 * total_hits / total_observations,
        "average_candidates_per_observation":
            total_candidates / total_observations,
        "total_candidate_rows": total_candidates,
        "total_observations": total_observations,
    })

    print(f"Candidate recall: {results[-1]['candidate_recall_pct']:.3f}%")

# Save consolidated results
import pandas as pd

df = pd.DataFrame(results)
df.to_csv(OUTPUT / "radius_comparison.csv", index=False)

with (OUTPUT / "radius_comparison.json").open("w") as f:
    json.dump(results, f, indent=4)

print("\nFINAL RADIUS COMPARISON")
print(df.to_string(index=False))
print("\nSaved results:", OUTPUT)
