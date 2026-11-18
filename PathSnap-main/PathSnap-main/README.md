# PathSnap: Distributed Map-Matching & Route Reconstruction

> **PES University ML Hackathon 2026** — Team Submission

## Overview

PathSnap is a **distributed two-laptop system** for GPS map-matching and route reconstruction on road networks. It combines:
- **5 trained ML classifiers** evaluated via 3-fold cross-validation
- **Bayesian hyperparameter optimisation** (Optuna TPE, 20 trials)
- **HMM + Viterbi decoder** for topologically coherent route reconstruction
- **FastAPI backend** (Laptop 1) + **Bootstrap web UI** (Laptop 2)

---

## Repository Structure

```
PathSnap/
├── src/
│   ├── dataset.py          # Data loading, projection, candidate generation
│   ├── features.py         # 12-feature extraction per GPS-arc pair
│   ├── models.py           # 5 ML model factory + MODEL_REGISTRY
│   ├── hmm_decoder.py      # HMM Viterbi log-space sequential decoder
│   ├── metrics.py          # Top-1 accuracy, Macro F1, continuity, log loss
│   ├── visualize.py        # Folium interactive HTML map generator
│   └── server.py           # FastAPI server (Laptop 1)
├── artifacts/
│   ├── models/             # Trained .joblib model files
│   ├── results/            # CSV experiment results
│   └── plots/              # EDA, model, Bayesian, HMM, map plots
├── templates/
│   └── index.html          # Evaluator dashboard (Laptop 2 UI)
├── reports/
│   └── PathSnap_Technical_Report.pdf
├── run_eda.py                            # EDA pipeline
├── train_and_evaluate_models.py          # 5-model training + CV + test eval
├── run_bayesian_optimization_pipeline.py # Optuna Bayesian search
├── run_hmm_evaluation.py                 # HMM Viterbi evaluation
├── generate_folium_maps.py               # Folium HTML map generation
├── run_evaluator_test_demo.py            # Full evaluator_test inference demo
├── run_ui.py                             # Laptop 2 simple HTTP server
├── requirements.txt
└── .gitignore
```

---

## Key Results (Evaluator Test Set — 10 Trips)

| Metric | Classifier Only | HMM + Viterbi |
|--------|:--------------:|:-------------:|
| Top-1 Accuracy | 81.31% | 77.75% |
| Route Continuity | 89.00% | **99.10%** |
| Continuity Gain | — | **+10.09%** |

> HMM Viterbi trades minor pointwise accuracy for dramatically better topological route coherence (+10% continuity).

### Model Comparison (5 Models, Student Test Set)

| Model | CV Macro F1 | Top-1 Acc | Route Continuity | Log Loss |
|-------|:-----------:|:---------:|:----------------:|:--------:|
| Logistic Regression | 0.8700 | 81.04% | 89.11% | 0.1161 |
| KNN (k=15) | 0.8664 | 79.51% | 87.72% | 0.3166 |
| Decision Tree | 0.8706 | 80.16% | 87.92% | 0.2255 |
| Random Forest | 0.8837 | 81.90% | 89.54% | 0.0920 |
| **HistGradBoost (Tuned)** | **0.8857** | **81.94%** | **89.34%** | **0.0919** |

---

## Installation

```bash
# Clone repository
git clone <your-github-url>
cd PathSnap

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Full Pipeline (Laptop 1)

```bash
# Step 1: EDA
python run_eda.py

# Step 2: Train 5 models + evaluate
python train_and_evaluate_models.py

# Step 3: Bayesian optimisation (20 trials)
python run_bayesian_optimization_pipeline.py

# Step 4: HMM Viterbi evaluation
python run_hmm_evaluation.py

# Step 5: Generate Folium maps
python generate_folium_maps.py

# Step 6: Run evaluator test demo
python run_evaluator_test_demo.py
```

---

## Two-Laptop Deployment

### Laptop 1 — API Server

Place the entire project folder (including `artifacts/` and `evaluator_test/`) on Laptop 1.

```bash
# Start the FastAPI server on port 8000
python src/server.py
# OR
uvicorn src.server:app --host 0.0.0.0 --port 8000
```

The server will log:
```
[INFO] [PathSnap-API] Loading trained model artifact from: artifacts/models/gradient_boosting_tuned.joblib
[INFO] [PathSnap-API] Model and HMM Viterbi decoder loaded successfully!
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Find Laptop 1's local IP:
```bash
ipconfig   # Windows → look for IPv4 Address e.g. 192.168.1.10
```

### Laptop 2 — Evaluator Dashboard

Open a browser on Laptop 2 and navigate to:

```
http://<Laptop1-IP>:8000/ui
```

Or serve the UI locally:
```bash
python run_ui.py
# Then open http://localhost:8080 and set API URL to http://192.168.1.10:8000
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | System info & status |
| `GET` | `/health` | Model load status |
| `GET` | `/trips` | List all 10 evaluator_test trips |
| `POST` | `/predict/{trip_id}` | Full ML + HMM inference + accuracy |
| `GET` | `/map/{trip_id}` | Interactive Folium HTML map |
| `GET` | `/ui` | Evaluator dashboard (served from Laptop 1) |

### Example API Response (`/predict/00000049`)

```json
{
  "status": "success",
  "trip_id": "00000049",
  "gps_observation_count": 2642,
  "pointwise_top1_accuracy": 94.81,
  "route_continuity": 99.73,
  "classifier_top1_accuracy": 96.18,
  "classifier_continuity": 96.40,
  "continuity_gain": 3.33,
  "candidate_miss_rate": 0.0,
  "model_inference_ms": 182.4,
  "hmm_decoding_ms": 1240.6,
  "total_latency_ms": 4199.0,
  "sample_gps_coords": [[49.84, 8.59], ...],
  "predicted_route_coords": [[49.84, 8.59], ...]
}
```

---

## Dataset Structure

```
evaluator_test/
└── evaluator_test/
    ├── 00000015/
    │   ├── 00000015.track   # lon lat timestamp (1Hz GPS)
    │   ├── 00000015.nodes   # lon lat (road network nodes)
    │   ├── 00000015.arcs    # from_node to_node (directed road segments)
    │   └── 00000015.route   # ground-truth arc IDs (server-side only)
    ├── 00000017/ ...
    └── 00000094/
```

---

## Architecture

```
Laptop 2 (Evaluator)                    Laptop 1 (Server — LAN/Wi-Fi)
┌──────────────────────┐                ┌────────────────────────────────┐
│  Bootstrap Web UI    │                │  FastAPI Application           │
│  index.html          │                │                                │
│                      │  POST          │  /predict/{trip_id}            │
│  Select trip ID ─────┼───────────────►│  1. load_trip_data()           │
│  (dropdown)          │  /predict/     │  2. project_coordinates()      │
│                      │  00000049      │  3. generate_candidates()      │
│  Show metrics        │◄───────────────┤  4. extract_features()         │
│  Show Folium map     │  JSON response │  5. model.predict_proba()      │
│                      │                │  6. HMM Viterbi decode()       │
│                      │                │  7. compute_metrics()          │
└──────────────────────┘                │  (ground truth stays server)   │
                                        └────────────────────────────────┘
```

---

## AI Use Disclosure

Python code was developed with AI assistance for boilerplate (data loading, feature engineering scaffolding, FastAPI routing). **All model training, evaluation, Bayesian optimisation trials, and HMM Viterbi experiments were executed on real data.** All reported metrics are from actual experiment runs. Report analysis, interpretation, and conclusions are the team's own work.

