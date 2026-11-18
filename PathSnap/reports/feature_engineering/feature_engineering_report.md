# Candidate Generation and Feature Engineering Report

## 1. Objective

The objective of candidate generation is to identify nearby road
segments for every GPS observation.

Instead of evaluating the entire road network for every GPS point,
we restrict the search to nearby road segments.

## 2. Candidate Generation

A search radius of 50 metres was used.

The implementation uses:

- Shapely for geometric operations.
- STRtree for spatial indexing.
- Projected coordinates for distance calculations in metres.

For each GPS observation, nearby road segments are identified
and stored as candidate road segments.

## 3. Feature Engineering

Six main features were extracted:

1. distance_m:
   Distance between the GPS observation and candidate road.

2. bearing_difference_deg:
   Difference between GPS movement direction and road direction.

3. road_length_m:
   Length of the candidate road segment.

4. projection_fraction:
   Relative position of the projected GPS point along the road segment.

5. gps_speed_mps:
   Estimated GPS movement speed.

6. time_delta_s:
   Time difference between consecutive GPS observations.

## 4. Training Labels

The reference route is used to derive binary candidate labels.

- Positive: Candidate has the same directed endpoints as the
  nearest reference-route arc.
- Negative: Candidate does not match the reference arc.

Ground-truth-derived metadata is excluded from the model input features
to prevent data leakage.

## 5. Candidate Generation Results

Across 80 training trips:

- GPS observations: 208,417
- Candidate records generated: 1,866,677
- Positive candidates: 185,862
- Negative candidates: 1,680,815
- Weighted reference candidate recall: 89.17%

## 6. Final Feature Dataset

After feature processing and quality filtering:

- Feature records: 1,831,153
- Accepted GPS observations: 185,841
- Training trips: 80

## 7. Limitations

The current reference-label generation uses independent nearest
reference-arc matching for each GPS observation.

A 50-metre reference-distance threshold is applied when selecting
eligible observations.

Some observations do not have their reference road among the generated
candidates. These cases cannot be correctly classified by the current
candidate classifier.

## 8. Outcome

The final feature dataset was prepared for the five machine-learning
models used in the PathSnap model comparison experiments.
