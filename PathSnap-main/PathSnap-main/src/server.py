"""
server.py - FastAPI Backend for PathSnap Map-Matching System (Laptop 1).
Hosts the trained model, HMM Viterbi decoder, server-side reference data,
and provides evaluation endpoints with detailed live console logging.
"""

import glob
import logging
import os
import sys
import time
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.dataset import (
    align_ground_truth_route,
    build_spatial_indices,
    generate_candidates,
    load_trip_data,
    project_coordinates,
)
from src.features import extract_features_for_trip
from src.hmm_decoder import HMMRouteDecoder
from src.metrics import calculate_route_continuity, compute_observation_metrics
from src.visualize import create_folium_map

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [PathSnap-API] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("pathsnap_server")

app = FastAPI(
    title="PathSnap Distributed Map-Matching API",
    description="Backend API hosting the machine learning classifier, HMM Viterbi sequential decoder, and server-side evaluation.",
    version="2.0.0",
)

# Enable CORS for multi-laptop deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model and decoder state
MODEL_PATH = "artifacts/models/gradient_boosting_tuned.joblib"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "artifacts/models/gradient_boosting.joblib"

logger.info(f"Loading trained model artifact from: {MODEL_PATH}")
trained_model = joblib.load(MODEL_PATH)
decoder = HMMRouteDecoder(sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5)
logger.info("Model and HMM Viterbi decoder loaded successfully!")

# Determine available evaluation trips
def get_eval_directory():
    # Priority 1: nested evaluator_test/evaluator_test (extracted zip structure)
    if os.path.exists("evaluator_test/evaluator_test"):
        return "evaluator_test/evaluator_test"
    # Priority 2: evaluator_test at root (flat structure)
    elif os.path.exists("evaluator_test"):
        dirs = [d for d in os.listdir("evaluator_test") if os.path.isdir(os.path.join("evaluator_test", d)) and d.isdigit()]
        if dirs:
            return "evaluator_test"
    # Priority 3: extracted zip evaluator_test
    if os.path.exists("Student folder_extracted/Student folder/evaluator_test"):
        return "Student folder_extracted/Student folder/evaluator_test"
    # Priority 4: student_test fallback
    elif os.path.exists("Student folder_extracted/Student folder/student_test"):
        return "Student folder_extracted/Student folder/student_test"
    return None

EVAL_DIR = get_eval_directory()


@app.get("/")
def root():
    return {
        "system": "PathSnap Distributed Map-Matching & Route Reconstruction API",
        "status": "online",
        "eval_trips_available": len(glob.glob(f"{EVAL_DIR}/*")) if EVAL_DIR else 0,
        "endpoints": ["/health", "/trips", "/predict/{trip_id}", "/map/{trip_id}", "/ui"],
    }


@app.get("/ui", response_class=HTMLResponse)
def get_evaluator_ui():
    """
    Serves the Evaluator Dashboard (Laptop 2 Web Interface).
    """
    ui_path = "templates/index.html"
    if os.path.exists(ui_path):
        with open(ui_path, "r", encoding="utf-8") as f:
            return f.read()
    raise HTTPException(status_code=404, detail="UI template not found.")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": trained_model is not None,
        "model_type": type(trained_model).__name__,
        "evaluation_dataset_dir": EVAL_DIR,
        "timestamp": time.time(),
    }


@app.get("/trips")
def list_evaluation_trips():
    """
    Returns the list of the 10 reserved evaluation trips available on the server.
    """
    if not EVAL_DIR or not os.path.exists(EVAL_DIR):
        raise HTTPException(status_code=500, detail="Evaluation trips directory not found on server.")

    trip_paths = sorted(glob.glob(f"{EVAL_DIR}/*"))
    trip_ids = [os.path.basename(p) for p in trip_paths if os.path.isdir(p)]

    logger.info(f"Listed {len(trip_ids)} evaluation trips for client request.")
    return {
        "count": len(trip_ids),
        "trips": trip_ids,
        "source": os.path.basename(EVAL_DIR),
    }


@app.post("/predict/{trip_id}")
@app.get("/predict/{trip_id}")
def evaluate_trip(trip_id: str):
    """
    Core Inference & Evaluation Endpoint:
    Processes the selected trip through candidate generation, feature extraction,
    model probability prediction, and HMM Viterbi route reconstruction.
    Calculates accuracy using server-side reference data.
    """
    t_start = time.time()
    logger.info(f"=== INFERENCE REQUEST RECEIVED FOR TRIP: {trip_id} ===")

    trip_dir = os.path.join(EVAL_DIR, trip_id)
    if not os.path.exists(trip_dir):
        logger.error(f"Trip ID '{trip_id}' not found in {EVAL_DIR}")
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found on server.")

    try:
        # 1. Load raw trip files
        trip_data = load_trip_data(trip_dir)
        num_obs = len(trip_data["track"])
        logger.info(f"Loaded {num_obs} GPS points and {len(trip_data['arcs'])} road network arcs.")

        # 2. Local spatial projection
        nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])

        # 3. Server-side reference alignment (kept strictly server-side)
        assigned_arcs = None
        if trip_data["route"] is not None:
            assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)

        # 4. Spatial Candidate Generation
        spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
        candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)
        logger.info(f"Spatial candidates generated for {num_obs} GPS observations.")

        # 5. Feature Extraction
        X, _, meta = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)

        # 6. Classifier Inference
        t_model = time.time()
        probs = trained_model.predict_proba(X)[:, 1]
        model_time = (time.time() - t_model) * 1000.0

        # Classifier-only prediction
        obs_clf = compute_observation_metrics(meta, [1 if cid == assigned_arcs[pt] else 0 for pt, cid in meta] if assigned_arcs is not None else None, probs, trip_data["arcs"])
        clf_acc = obs_clf["top1_accuracy"] * 100.0
        clf_cont = obs_clf["route_continuity"] * 100.0

        # 7. HMM Viterbi Sequential Route Reconstruction
        t_hmm = time.time()
        ml_probs_per_point = []
        cursor = 0
        for i in range(num_obs):
            k = len(candidates[i])
            ml_probs_per_point.append(probs[cursor : cursor + k])
            cursor += k

        time_deltas = np.diff(trip_data["track"][:, 2]) if num_obs > 1 else None
        hmm_seq = decoder.decode(candidates, ml_probs_per_point, trip_data["arcs"], nodes_m, track_m, time_deltas)
        hmm_time = (time.time() - t_hmm) * 1000.0

        # 8. Evaluation against server-side reference data
        if assigned_arcs is not None:
            correct_count = sum(1 for i in range(num_obs) if hmm_seq[i] == assigned_arcs[i])
            hmm_acc = (correct_count / num_obs) * 100.0
        else:
            hmm_acc = None

        hmm_cont = calculate_route_continuity(hmm_seq, trip_data["arcs"]) * 100.0
        total_time_ms = (time.time() - t_start) * 1000.0

        logger.info(f"SUCCESS [Trip {trip_id}]: Clf Acc={clf_acc:.2f}%, HMM Acc={hmm_acc:.2f}% | Clf Cont={clf_cont:.2f}%, HMM Cont={hmm_cont:.2f}% | Latency: {total_time_ms:.1f}ms")

        # Prepare route coordinates for visualization
        pred_coords = []
        for arc_id in hmm_seq:
            if 0 <= arc_id < len(trip_data["arcs"]):
                u, v = trip_data["arcs"][arc_id]
                pred_coords.append([float(trip_data["nodes"][u, 1]), float(trip_data["nodes"][u, 0])])

        raw_gps_coords = [[float(pt[1]), float(pt[0])] for pt in trip_data["track"]]

        return {
            "status": "success",
            "trip_id": trip_id,
            "gps_observation_count": num_obs,
            "pointwise_top1_accuracy": round(hmm_acc, 2) if hmm_acc is not None else "N/A",
            "route_continuity": round(hmm_cont, 2),
            "classifier_top1_accuracy": round(clf_acc, 2),
            "classifier_continuity": round(clf_cont, 2),
            "continuity_gain": round(hmm_cont - clf_cont, 2),
            "candidate_miss_rate": round(obs_clf["candidate_miss_rate"] * 100.0, 2),
            "model_inference_ms": round(model_time, 1),
            "hmm_decoding_ms": round(hmm_time, 1),
            "total_latency_ms": round(total_time_ms, 1),
            "sample_gps_coords": raw_gps_coords[::max(1, len(raw_gps_coords) // 200)],  # sampled for UI map
            "predicted_route_coords": pred_coords[::max(1, len(pred_coords) // 200)],
        }

    except Exception as e:
        logger.error(f"Error processing trip {trip_id}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/map/{trip_id}", response_class=HTMLResponse)
def get_trip_map(trip_id: str):
    """
    Returns an interactive Folium HTML map for the specified trip.
    """
    map_path = f"artifacts/maps/trip_{trip_id}.html"
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            return f.read()

    # If not pre-generated, generate on demand
    trip_dir = os.path.join(EVAL_DIR, trip_id)
    if not os.path.exists(trip_dir):
        raise HTTPException(status_code=404, detail="Trip not found")

    trip_data = load_trip_data(trip_dir)
    nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
    assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
    spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
    candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)

    X, _, meta = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)
    probs = trained_model.predict_proba(X)[:, 1]

    ml_probs = []
    cursor = 0
    for i in range(len(track_m)):
        k = len(candidates[i])
        ml_probs.append(probs[cursor : cursor + k])
        cursor += k

    hmm_seq = decoder.decode(candidates, ml_probs, trip_data["arcs"], nodes_m, track_m)
    obs_clf = compute_observation_metrics(meta, None, probs, trip_data["arcs"])

    fmap = create_folium_map(
        track=trip_data["track"],
        arcs=trip_data["arcs"],
        nodes=trip_data["nodes"],
        true_route_arcs=assigned_arcs,
        classifier_pred_arcs=obs_clf["pred_sequence"],
        hmm_pred_arcs=hmm_seq,
        trip_id=trip_id,
        out_html=map_path,
    )
    with open(map_path, "r", encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    import uvicorn
    # Bind to 0.0.0.0 so Laptop 2 on local Wi-Fi / LAN can connect directly
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)
