"""
run_hmm_evaluation.py - Evaluates HMM Viterbi Route Reconstruction
against Classifier-Only predictions across all 10 Student-Testing trips.
Computes Top-1 pointwise accuracy, route continuity, and generates comparative plots.
"""

import glob
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

os.makedirs("artifacts/results", exist_ok=True)
os.makedirs("artifacts/plots/hmm", exist_ok=True)

test_dirs = sorted(glob.glob("Student folder_extracted/Student folder/student_test/*"))
print(f"Loading tuned model and evaluating HMM on {len(test_dirs)} student-test trips...")

tuned_model = joblib.load("artifacts/models/gradient_boosting_tuned.joblib")
decoder = HMMRouteDecoder(sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5)

trip_eval_results = []
trip_trajectories = {}

for d in test_dirs:
    trip_id = os.path.basename(d)
    t0 = time.time()
    trip_data = load_trip_data(d)
    nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
    assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
    spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
    candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)

    # Extract features
    X_test, _, meta_test = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)

    # ML Inference
    t_infer_start = time.time()
    probs = tuned_model.predict_proba(X_test)[:, 1]
    infer_ms = (time.time() - t_infer_start) * 1000.0

    # 1. Classifier-Only Prediction
    obs_clf = compute_observation_metrics(meta_test, [1 if cid == assigned_arcs[pt] else 0 for pt, cid in meta_test], probs, trip_data["arcs"])
    clf_seq = obs_clf["pred_sequence"]
    clf_top1 = obs_clf["top1_accuracy"]
    clf_cont = obs_clf["route_continuity"]

    # Format probabilities per observation for HMM
    ml_probs_per_point = []
    idx_cursor = 0
    for i in range(len(track_m)):
        k_cands = len(candidates[i])
        ml_probs_per_point.append(probs[idx_cursor : idx_cursor + k_cands])
        idx_cursor += k_cands

    # 2. Classifier + HMM Viterbi Route Reconstruction
    t_hmm_start = time.time()
    time_deltas = np.diff(trip_data["track"][:, 2]) if len(trip_data["track"]) > 1 else None
    hmm_seq = decoder.decode(candidates, ml_probs_per_point, trip_data["arcs"], nodes_m, track_m, time_deltas)
    hmm_ms = (time.time() - t_hmm_start) * 1000.0

    # Evaluate HMM Sequence
    hmm_correct = sum(1 for i in range(len(track_m)) if hmm_seq[i] == assigned_arcs[i])
    hmm_top1 = hmm_correct / len(track_m)
    hmm_cont = calculate_route_continuity(hmm_seq, trip_data["arcs"])

    total_proc_time = time.time() - t0

    trip_eval_results.append({
        "Trip ID": trip_id,
        "GPS Observations": len(track_m),
        "Classifier Top-1 (%)": round(clf_top1 * 100, 2),
        "HMM Top-1 (%)": round(hmm_top1 * 100, 2),
        "Top-1 Change (%)": round((hmm_top1 - clf_top1) * 100, 2),
        "Classifier Continuity (%)": round(clf_cont * 100, 2),
        "HMM Continuity (%)": round(hmm_cont * 100, 2),
        "Continuity Improvement (%)": round((hmm_cont - clf_cont) * 100, 2),
        "HMM Time (ms)": round(hmm_ms, 1),
        "Total Time (s)": round(total_proc_time, 2),
    })

    trip_trajectories[trip_id] = {
        "trip_data": trip_data,
        "track_m": track_m,
        "assigned_arcs": assigned_arcs,
        "clf_seq": clf_seq,
        "hmm_seq": hmm_seq,
    }

    print(f"Trip {trip_id}: Clf Acc={clf_top1*100:.2f}%, HMM Acc={hmm_top1*100:.2f}% | Clf Cont={clf_cont*100:.2f}%, HMM Cont={hmm_cont*100:.2f}% (+{(hmm_cont-clf_cont)*100:.2f}%)")

df_hmm = pd.DataFrame(trip_eval_results)
df_hmm.to_csv("artifacts/results/hmm_vs_classifier_comparison.csv", index=False)

print("\n=== HMM VITERBI ROUTE RECONSTRUCTION EVALUATION TABLE ===")
print(df_hmm.to_string(index=False))

# Aggregate summary
mean_clf_acc = df_hmm["Classifier Top-1 (%)"].mean()
mean_hmm_acc = df_hmm["HMM Top-1 (%)"].mean()
mean_clf_cont = df_hmm["Classifier Continuity (%)"].mean()
mean_hmm_cont = df_hmm["HMM Continuity (%)"].mean()

print(f"\nOverall Summary Across 10 Test Trips:")
print(f"Pointwise Accuracy: Classifier={mean_clf_acc:.2f}% -> Classifier+HMM={mean_hmm_acc:.2f}% (Change: {mean_hmm_acc - mean_clf_acc:+.2f}%)")
print(f"Route Continuity:   Classifier={mean_clf_cont:.2f}% -> Classifier+HMM={mean_hmm_cont:.2f}% (Gain: +{mean_hmm_cont - mean_clf_cont:.2f}%)")

# Save summary
summary_df = pd.DataFrame([
    {
        "Metric": "Pointwise Top-1 Accuracy",
        "Classifier-Only": f"{mean_clf_acc:.2f}%",
        "Classifier + HMM": f"{mean_hmm_acc:.2f}%",
        "Net Improvement": f"{mean_hmm_acc - mean_clf_acc:+.2f}%",
    },
    {
        "Metric": "Route Continuity",
        "Classifier-Only": f"{mean_clf_cont:.2f}%",
        "Classifier + HMM": f"{mean_hmm_cont:.2f}%",
        "Net Improvement": f"+{mean_hmm_cont - mean_clf_cont:.2f}%",
    },
])
summary_df.to_csv("artifacts/results/hmm_summary_metrics.csv", index=False)

# Comparative Visualizations
# 1. Continuity Gain Bar Chart per trip
plt.figure(figsize=(10, 5))
x = np.arange(len(df_hmm))
width = 0.35
plt.bar(x - width/2, df_hmm["Classifier Continuity (%)"], width, label="Classifier-Only Continuity", color="#E53935")
plt.bar(x + width/2, df_hmm["HMM Continuity (%)"], width, label="Classifier + HMM Continuity", color="#1E88E5")
plt.ylabel("Route Continuity (%)")
plt.xlabel("Student Test Trip ID")
plt.title("Route Continuity Enhancement: Classifier-Only vs Classifier + HMM")
plt.xticks(x, df_hmm["Trip ID"], rotation=25)
plt.ylim(70, 103)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("artifacts/plots/hmm/continuity_comparison.png", dpi=300)
plt.close()

# 2. Pointwise Accuracy Comparison per trip
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, df_hmm["Classifier Top-1 (%)"], width, label="Classifier-Only Top-1", color="#FB8C00")
plt.bar(x + width/2, df_hmm["HMM Top-1 (%)"], width, label="Classifier + HMM Top-1", color="#43A047")
plt.ylabel("Pointwise Accuracy (%)")
plt.xlabel("Student Test Trip ID")
plt.title("Pointwise Top-1 Accuracy: Classifier-Only vs Classifier + HMM")
plt.xticks(x, df_hmm["Trip ID"], rotation=25)
plt.ylim(65, 100)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("artifacts/plots/hmm/accuracy_comparison.png", dpi=300)
plt.close()

print("\nHMM evaluation and plots completed successfully!")

