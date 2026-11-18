
# PathSnap — GPS Trajectory Map Matching & Route Reconstruction

**Team 11 | Machine Learning Hackathon**

PathSnap is a machine learning-based GPS trajectory map-matching system that reconstructs realistic vehicle routes from noisy GPS observations using road-network information.

The system combines machine learning-based road candidate classification with Hidden Markov Models (HMM) and Viterbi decoding to identify the most likely sequence of road segments travelled.

## Problem Statement

GPS measurements can be inaccurate due to urban buildings, signal interference, multipath effects, GPS dropouts and irregular sampling frequencies.

PathSnap aims to:
- Process noisy GPS trajectories.
- Identify candidate road segments.
- Extract meaningful geographical and movement features.
- Train and compare multiple machine learning models.
- Reconstruct continuous routes using HMM and Viterbi.
- Expose predictions through a FastAPI backend.
- Provide an evaluator-facing interface.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Hyperparameter Optimisation | Optuna |
| Geospatial Processing | Shapely, PyProj |
| Road Network | NetworkX |
| Visualisation | Matplotlib, Folium |
| Backend | FastAPI |
| API Server | Uvicorn |
| Frontend | To be finalised |

## System Architecture

The system consists of the following components:

1. Dataset inspection and preprocessing.
2. GPS-to-road candidate generation.
3. Feature engineering.
4. Model training and comparison.
5. Bayesian hyperparameter optimisation.
6. HMM-based route reconstruction.
7. Viterbi sequence decoding.
8. FastAPI prediction backend.
9. Evaluator-facing frontend.
10. Two-laptop deployment.

## Dataset

The dataset consists of GPS trajectories and corresponding road-network information.

Each trip contains:

- `.track` — GPS longitude, latitude and timestamp.
- `.nodes` — Road-network geographical nodes.
- `.arcs` — Connections between road-network nodes.
- `.route` — Reference sequence of road-segment IDs.

Dataset partitions specified by the hackathon:

| Partition | Trips | Usage |
|---|---:|---|
| Training | 80 | Training and cross-validation |
| Student Testing | 10 | Development-stage evaluation |
| Evaluator Testing | 10 | Final evaluation only |

Raw datasets are maintained locally and are not committed to this repository.

Dataset paths are configured through `PathSnap/configs/data.yaml`.

## Project Structure

- `PathSnap/src/` — Core Python implementation.
- `PathSnap/configs/` — Project configurations.
- `PathSnap/scripts/` — Execution scripts.
- `PathSnap/notebooks/` — Exploratory analysis.
- `PathSnap/tests/` — Automated tests.
- `PathSnap/artifacts/` — Trained models and inference artifacts.
- `PathSnap/experiments/` — Experiment logs and results.
- `PathSnap/reports/` — Graphs, maps and technical report.
- `PathSnap/frontend/` — Evaluator interface.
- `PathSnap/deployment/` — Deployment utilities.

## Installation

Python 3.10 or later is recommended.

Clone the repository:

```bash
git clone <repository-url>
cd ML-Hackathon
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Configure the dataset paths in:

```text
PathSnap/configs/data.yaml
```

## Evaluation and Integrity

- At least five distinct models must be trained and evaluated.
- Model selection must be supported by actual experimental results.
- Bayesian hyperparameter optimisation must be demonstrated.
- Classifier-only and HMM-decoded predictions must be compared.
- Evaluator ground-truth labels must remain server-side.
- Evaluator-testing data must not be used during development.
- All reported metrics and visualisations must be reproducible.

## Distributed Deployment

**Laptop 1 — Backend**
- Trained machine learning model.
- HMM and Viterbi decoder.
- Road-network information.
- FastAPI server.

**Laptop 2 — Frontend**
- Evaluator interface.
- Trip selection.
- API communication.
- Predicted route and evaluation display.

## Project Status

Currently under development.

Implementation, experimental results and documentation will be updated as the hackathon progresses.
