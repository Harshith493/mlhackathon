"""
generate_all_evaluator_maps.py
==============================
Generates interactive Folium HTML maps for all 10 evaluator_test trips
and static PNG snapshots for embedding in the report.
"""

import os, sys
sys.path.insert(0, ".")

from src.dataset import (
    align_ground_truth_route, build_spatial_indices,
    generate_candidates, load_trip_data, project_coordinates,
)
from src.features import extract_features_for_trip
from src.hmm_decoder import HMMRouteDecoder
from src.metrics import compute_observation_metrics
from src.visualize import create_folium_map
import joblib, numpy as np

EVAL_DIR = "evaluator_test/evaluator_test"
MODEL_PATH = "artifacts/models/gradient_boosting_tuned.joblib"
os.makedirs("artifacts/maps", exist_ok=True)

print("Loading model...", end=" ", flush=True)
model = joblib.load(MODEL_PATH)
decoder = HMMRouteDecoder(sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5)
print("OK!")

trip_ids = sorted(os.listdir(EVAL_DIR))
print(f"Generating maps for {len(trip_ids)} trips: {trip_ids}\n")

for tid in trip_ids:
    trip_dir = os.path.join(EVAL_DIR, tid)
    out_html = f"artifacts/maps/trip_{tid}.html"
    if os.path.exists(out_html):
        print(f"  [SKIP] {tid} — map already exists")
        continue
    try:
        trip_data = load_trip_data(trip_dir)
        nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
        assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
        spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
        candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)
        X, _, meta = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)
        probs = model.predict_proba(X)[:, 1]
        labels = [1 if cid == assigned_arcs[pt] else 0 for pt, cid in meta]
        obs_clf = compute_observation_metrics(meta, labels, probs, trip_data["arcs"])

        ml_probs = []
        cursor = 0
        for i in range(len(track_m)):
            k = len(candidates[i])
            ml_probs.append(probs[cursor:cursor+k])
            cursor += k
        time_deltas = np.diff(trip_data["track"][:, 2]) if len(track_m) > 1 else None
        hmm_seq = decoder.decode(candidates, ml_probs, trip_data["arcs"], nodes_m, track_m, time_deltas)

        create_folium_map(
            track=trip_data["track"],
            arcs=trip_data["arcs"],
            nodes=trip_data["nodes"],
            true_route_arcs=assigned_arcs,
            classifier_pred_arcs=obs_clf["pred_sequence"],
            hmm_pred_arcs=hmm_seq,
            trip_id=tid,
            out_html=out_html,
        )
        print(f"  [OK] {tid} -> {out_html}")
    except Exception as e:
        print(f"  [ERR] {tid}: {e}")

print("\nAll evaluator_test maps generated!")

