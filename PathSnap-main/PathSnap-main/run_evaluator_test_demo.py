"""
run_evaluator_test_demo.py
==========================
Runs the full PathSnap inference pipeline on all 10 evaluator_test trips
and outputs a formatted evaluation report + saves results CSV.

This simulates exactly what the FastAPI server does on each /predict/{trip_id}
API request but without needing a live HTTP connection — useful for verification.
"""

import json
import os
import sys
import time
import joblib
import numpy as np
import csv

sys.path.insert(0, ".")

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

EVAL_DIR = "evaluator_test/evaluator_test"
if not os.path.exists(EVAL_DIR):
    EVAL_DIR = "evaluator_test"
MODEL_PATH = "artifacts/models/gradient_boosting_tuned.joblib"
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "artifacts/models/gradient_boosting.joblib"

print("=" * 70)
print("  PathSnap — Evaluator Test Set Inference Report")
print("=" * 70)
print(f"Model     : {MODEL_PATH}")
print(f"Dataset   : {EVAL_DIR}/")
print(f"Timestamp : {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Load model
print("\nLoading model...", end=" ", flush=True)
model = joblib.load(MODEL_PATH)
decoder = HMMRouteDecoder(sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5)
print("OK!")

trip_dirs = sorted([
    d for d in os.listdir(EVAL_DIR)
    if os.path.isdir(os.path.join(EVAL_DIR, d))
])

print(f"\nFound {len(trip_dirs)} evaluator_test trips: {trip_dirs}\n")

results = []

header = f"{'Trip ID':<12} {'GPS pts':>7} {'Clf Acc':>8} {'Clf Cont':>9} {'HMM Acc':>8} {'HMM Cont':>9} {'Cont Gain':>10} {'Latency':>9}"
sep = "-" * len(header)
print(header)
print(sep)

for trip_id in trip_dirs:
    t_start = time.time()
    trip_dir = os.path.join(EVAL_DIR, trip_id)

    try:
        # Load data
        trip_data = load_trip_data(trip_dir)
        num_obs = len(trip_data["track"])

        # Project
        nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])

        # Ground truth
        assigned_arcs = None
        if trip_data["route"] is not None:
            assigned_arcs, _ = align_ground_truth_route(
                track_m, trip_data["route"], trip_data["arcs"], nodes_m
            )

        # Candidates
        spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
        candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)

        # Features
        X, _, meta = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)

        # Classifier inference
        probs = model.predict_proba(X)[:, 1]

        labels = None
        if assigned_arcs is not None:
            labels = [1 if cid == assigned_arcs[pt] else 0 for pt, cid in meta]

        obs_clf = compute_observation_metrics(meta, labels, probs, trip_data["arcs"])
        clf_acc = obs_clf["top1_accuracy"] * 100.0
        clf_cont = obs_clf["route_continuity"] * 100.0

        # HMM Viterbi
        ml_probs_per_point = []
        cursor = 0
        for i in range(num_obs):
            k = len(candidates[i])
            ml_probs_per_point.append(probs[cursor: cursor + k])
            cursor += k

        time_deltas = np.diff(trip_data["track"][:, 2]) if num_obs > 1 else None
        hmm_seq = decoder.decode(candidates, ml_probs_per_point, trip_data["arcs"], nodes_m, track_m, time_deltas)

        if assigned_arcs is not None:
            hmm_acc = sum(1 for i in range(num_obs) if hmm_seq[i] == assigned_arcs[i]) / num_obs * 100.0
        else:
            hmm_acc = None

        hmm_cont = calculate_route_continuity(hmm_seq, trip_data["arcs"]) * 100.0
        total_ms = (time.time() - t_start) * 1000.0

        cont_gain = hmm_cont - clf_cont
        hmm_acc_str = f"{hmm_acc:.2f}%" if hmm_acc is not None else "N/A"
        print(f"{trip_id:<12} {num_obs:>7} {clf_acc:>7.2f}% {clf_cont:>8.2f}% {hmm_acc:.2f}% {hmm_cont:>8.2f}% {cont_gain:>+9.2f}% {total_ms:>7.0f}ms")

        results.append({
            "trip_id": trip_id,
            "gps_obs": num_obs,
            "clf_top1_accuracy": round(clf_acc, 2),
            "clf_route_continuity": round(clf_cont, 2),
            "hmm_top1_accuracy": round(hmm_acc, 2) if hmm_acc is not None else None,
            "hmm_route_continuity": round(hmm_cont, 2),
            "continuity_gain": round(cont_gain, 2),
            "latency_ms": round(total_ms, 1),
        })

    except Exception as e:
        print(f"{trip_id:<12} ERROR: {e}")
        results.append({"trip_id": trip_id, "error": str(e)})

print(sep)

# Summary stats
valid = [r for r in results if "error" not in r]
if valid:
    avg_clf_acc = np.mean([r["clf_top1_accuracy"] for r in valid])
    avg_clf_cont = np.mean([r["clf_route_continuity"] for r in valid])
    avg_hmm_acc = np.mean([r["hmm_top1_accuracy"] for r in valid if r["hmm_top1_accuracy"] is not None])
    avg_hmm_cont = np.mean([r["hmm_route_continuity"] for r in valid])
    avg_gain = np.mean([r["continuity_gain"] for r in valid])
    avg_lat = np.mean([r["latency_ms"] for r in valid])

    print(f"\n{'AVERAGES':<12} {'':>7} {avg_clf_acc:>7.2f}% {avg_clf_cont:>8.2f}% {avg_hmm_acc:>7.2f}% {avg_hmm_cont:>8.2f}% {avg_gain:>+9.2f}% {avg_lat:>7.0f}ms")
    print()
    print(f"Key Finding: HMM Viterbi improves Route Continuity by {avg_gain:+.2f}% on average!")

# Save results
os.makedirs("artifacts/results", exist_ok=True)
out_csv = "artifacts/results/evaluator_test_results.csv"
if valid:
    keys = list(valid[0].keys())
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(valid)
    print(f"\nResults saved to: {out_csv}")

# Save JSON for the demo
out_json = "artifacts/results/evaluator_test_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"JSON saved to  : {out_json}")

print("\n" + "=" * 70)
print("  Evaluation Complete!")
print("=" * 70)
