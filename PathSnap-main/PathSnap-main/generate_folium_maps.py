"""
generate_folium_maps.py - Generates interactive Folium HTML maps
and high-resolution map views for evaluation trips.
"""

import os
import joblib
import numpy as np

from src.dataset import (
    align_ground_truth_route,
    build_spatial_indices,
    generate_candidates,
    load_trip_data,
    project_coordinates,
)
from src.features import extract_features_for_trip
from src.hmm_decoder import HMMRouteDecoder
from src.metrics import compute_observation_metrics
from src.visualize import create_folium_map

os.makedirs("artifacts/maps", exist_ok=True)
os.makedirs("artifacts/plots/maps", exist_ok=True)

tuned_model = joblib.load("artifacts/models/gradient_boosting_tuned.joblib")
decoder = HMMRouteDecoder(sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5)

target_trips = ["00000000", "00000007", "00000091"]

for trip_id in target_trips:
    trip_dir = f"Student folder_extracted/Student folder/student_test/{trip_id}"
    print(f"Generating interactive Folium map for Trip {trip_id}...")

    trip_data = load_trip_data(trip_dir)
    nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
    assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
    spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
    candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)

    X_test, _, meta_test = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)
    probs = tuned_model.predict_proba(X_test)[:, 1]

    # Classifier-only
    obs_clf = compute_observation_metrics(meta_test, [1 if cid == assigned_arcs[pt] else 0 for pt, cid in meta_test], probs, trip_data["arcs"])
    clf_seq = obs_clf["pred_sequence"]

    # HMM
    ml_probs_per_point = []
    cursor = 0
    for i in range(len(track_m)):
        k = len(candidates[i])
        ml_probs_per_point.append(probs[cursor : cursor + k])
        cursor += k

    time_deltas = np.diff(trip_data["track"][:, 2]) if len(trip_data["track"]) > 1 else None
    hmm_seq = decoder.decode(candidates, ml_probs_per_point, trip_data["arcs"], nodes_m, track_m, time_deltas)

    out_html = f"artifacts/maps/trip_{trip_id}.html"
    create_folium_map(
        track=trip_data["track"],
        arcs=trip_data["arcs"],
        nodes=trip_data["nodes"],
        true_route_arcs=assigned_arcs,
        classifier_pred_arcs=clf_seq,
        hmm_pred_arcs=hmm_seq,
        trip_id=trip_id,
        out_html=out_html,
    )
    print(f"Saved interactive Folium map: {out_html}")

    # Generate static 2D trajectory plot for PDF report embedding
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    # Plot ground truth route
    seen_gt = []
    for a in assigned_arcs:
        if not seen_gt or seen_gt[-1] != a:
            seen_gt.append(a)
    for idx, a in enumerate(seen_gt):
        u, v = trip_data["arcs"][a]
        label = "Ground Truth Route" if idx == 0 else ""
        plt.plot([trip_data["nodes"][u, 0], trip_data["nodes"][v, 0]],
                 [trip_data["nodes"][u, 1], trip_data["nodes"][v, 1]],
                 color="#2E7D32", linewidth=3.5, alpha=0.85, label=label)

    # Plot HMM reconstructed route
    seen_hmm = []
    for a in hmm_seq:
        if not seen_hmm or seen_hmm[-1] != a:
            seen_hmm.append(a)
    for idx, a in enumerate(seen_hmm):
        u, v = trip_data["arcs"][a]
        label = "HMM Reconstructed Route" if idx == 0 else ""
        plt.plot([trip_data["nodes"][u, 0], trip_data["nodes"][v, 0]],
                 [trip_data["nodes"][u, 1], trip_data["nodes"][v, 1]],
                 color="#00BCD4", linewidth=2.0, linestyle="--", alpha=0.9, label=label)

    # Plot raw GPS points
    plt.scatter(trip_data["track"][:, 0], trip_data["track"][:, 1],
                color="#1E88E5", s=10, alpha=0.6, label=f"Raw GPS Points (N={len(trip_data['track'])})")

    plt.xlabel("Longitude (deg)")
    plt.ylabel("Latitude (deg)")
    plt.title(f"PathSnap Route Reconstruction & Map-Matching: Trip {trip_id}")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/maps/map_snapshot_{trip_id}.png", dpi=300)
    plt.close()
    print(f"Saved static map view: artifacts/plots/maps/map_snapshot_{trip_id}.png")

print("\nFolium maps and static snapshots generated successfully!")

