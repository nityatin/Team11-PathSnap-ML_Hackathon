# Dataset Inspection and Preprocessing Report

## 1. Dataset Overview

The PathSnap dataset consists of individual GPS trips.
Each trip contains four files:

- .track: GPS longitude, latitude and timestamp.
- .nodes: Road network node coordinates.
- .arcs: Connections between road network nodes.
- .route: Chronological sequence of reference road arc IDs.

## 2. Dataset Partitioning

- Training trips: 80
- Student-test trips: 10
- Evaluator-test trips: 10

The evaluator-test partition was kept separate from development.

## 3. Inspection Performed

The dataset inspection process verified:

- Completeness of the four files for each trip.
- GPS coordinate validity.
- Timestamp consistency.
- Road network node and arc references.
- Reference route arc validity.
- Road network connectivity.
- Route continuity.

## 4. Inspection Results

- Trips inspected: 90
- Complete trips: 90
- Files expected: 360
- Files found: 360
- Malformed rows: 0

The inspection covered the training and student-test partitions.
Evaluator-test data was not inspected.

## 5. Data Quality Observations

Some trajectories contain timestamp gaps.
Some GPS observations are not closely aligned with their reference routes.
Some reference routes also contain minor continuity inconsistencies.

These observations were considered during subsequent candidate generation
and training-data preparation.

## 6. Outcome

The dataset inspection established that all 90 development trips
contained the expected files and no malformed rows were detected.

The validated dataset was then passed to the candidate-generation
and feature-engineering stages.
