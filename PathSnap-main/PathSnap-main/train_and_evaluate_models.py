"""
train_and_evaluate_models.py - Trains, cross-validates, and evaluates 5 machine learning
models across distinct algorithm families for PathSnap Map-Matching.
Records real metrics, saves model artifacts, and generates comparative visualizations.
"""

import glob
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

from src.dataset import (
    align_ground_truth_route,
    build_spatial_indices,
    generate_candidates,
    load_trip_data,
    project_coordinates,
)
from src.features import extract_features_for_trip
from src.metrics import (
    compute_classification_metrics,
    compute_observation_metrics,
)
from src.models import MODEL_REGISTRY, get_model

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

os.makedirs("artifacts/models", exist_ok=True)
os.makedirs("artifacts/results", exist_ok=True)
os.makedirs("artifacts/plots/models", exist_ok=True)

train_dirs = sorted(glob.glob("Student folder_extracted/Student folder/train/*"))
test_dirs = sorted(glob.glob("Student folder_extracted/Student folder/student_test/*"))

print(f"Preparing datasets: using 25 representative training trips and all 10 student-test trips...")

# Select 25 diverse training trips across trajectory lengths
selected_train_dirs = train_dirs[::3][:25]


def build_feature_dataset(trip_dirs, is_train=True):
    X_all = []
    y_all = []
    groups_all = []
    meta_all = []  # (trip_id, pt_idx, cid)
    arcs_by_trip = {}
    trips_data_store = {}

    for d in trip_dirs:
        trip_id = os.path.basename(d)
        trip_data = load_trip_data(d)
        nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
        assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
        spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])
        candidates = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=10)

        X, y, meta = extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs)

        X_all.append(X)
        y_all.append(y)
        groups_all.extend([trip_id] * len(y))
        meta_all.extend([(trip_id, m[0], m[1]) for m in meta])
        arcs_by_trip[trip_id] = trip_data["arcs"]
        trips_data_store[trip_id] = {
            "trip_data": trip_data,
            "nodes_m": nodes_m,
            "track_m": track_m,
            "candidates": candidates,
            "assigned_arcs": assigned_arcs,
        }

    X_mat = np.vstack(X_all)
    y_mat = np.concatenate(y_all)
    groups_mat = np.array(groups_all)
    return X_mat, y_mat, groups_mat, meta_all, arcs_by_trip, trips_data_store


print("Extracting features from training trips...")
t0 = time.time()
X_tr, y_tr, groups_tr, meta_tr, arcs_tr, _ = build_feature_dataset(selected_train_dirs, is_train=True)
print(f"Train dataset extracted: {X_tr.shape[0]} samples, {X_tr.shape[1]} features in {time.time()-t0:.2f}s")
print(f"Class distribution: Positive={np.sum(y_tr==1)} ({np.mean(y_tr==1)*100:.2f}%), Negative={np.sum(y_tr==0)}")

print("Extracting features from all 10 student-test trips...")
t1 = time.time()
X_te, y_te, groups_te, meta_te, arcs_te, test_store = build_feature_dataset(test_dirs, is_train=False)
print(f"Student-test dataset extracted: {X_te.shape[0]} samples in {time.time()-t1:.2f}s")

# Run 3-Fold Grouped Cross-Validation and Evaluation for each model
comparison_results = []
models_trained = {}
gkf = GroupKFold(n_splits=3)

for model_display_name, info in MODEL_REGISTRY.items():
    key = info["key"]
    family = info["family"]
    desc = info["desc"]
    print(f"\n=======================================================")
    print(f"Training and Evaluating: {model_display_name} ({family})")
    print(f"=======================================================")

    # 1. Cross-Validation on Train Data (Group-Aware)
    cv_macro_f1s = []
    cv_top1_accs = []

    for fold, (f_tr_idx, f_val_idx) in enumerate(gkf.split(X_tr, y_tr, groups=groups_tr)):
        model_fold = get_model(key)
        model_fold.fit(X_tr[f_tr_idx], y_tr[f_tr_idx])
        val_preds = model_fold.predict(X_tr[f_val_idx])
        val_probs = model_fold.predict_proba(X_tr[f_val_idx])[:, 1] if hasattr(model_fold, "predict_proba") else None

        f_metrics = compute_classification_metrics(y_tr[f_val_idx], val_preds, val_probs if val_probs is not None else val_preds)
        cv_macro_f1s.append(f_metrics["macro_f1"])

    mean_cv_macro_f1 = float(np.mean(cv_macro_f1s))
    print(f"Grouped CV Macro F1: {mean_cv_macro_f1:.4f} (Folds: {[round(x, 4) for x in cv_macro_f1s]})")

    # 2. Fit on full training data and measure training time
    final_model = get_model(key)
    train_start = time.time()
    final_model.fit(X_tr, y_tr)
    fit_time = time.time() - train_start

    # Save model artifact
    joblib.dump(final_model, f"artifacts/models/{key}.joblib")
    models_trained[key] = final_model

    # 3. Evaluate on full Student-Test trips
    infer_start = time.time()
    test_preds = final_model.predict(X_te)
    test_probs = final_model.predict_proba(X_te)[:, 1] if hasattr(final_model, "predict_proba") else np.array(test_preds, dtype=float)
    total_infer_time = (time.time() - infer_start) * 1000.0  # ms
    avg_infer_per_trip = total_infer_time / len(test_dirs)

    # Classification Metrics
    te_metrics = compute_classification_metrics(y_te, test_preds, test_probs)

    # Observation-Level Metrics across the 10 trips
    trip_top1_accs = []
    trip_conts = []
    trip_misses = []

    for tid in sorted(test_store.keys()):
        t_indices = [k for k, m in enumerate(meta_te) if m[0] == tid]
        t_sub_meta = [(meta_te[k][1], meta_te[k][2]) for k in t_indices]
        t_y = y_te[t_indices]
        t_probs = test_probs[t_indices]
        t_arcs = arcs_te[tid]

        obs_res = compute_observation_metrics(t_sub_meta, t_y, t_probs, t_arcs)
        trip_top1_accs.append(obs_res["top1_accuracy"])
        trip_conts.append(obs_res["route_continuity"])
        trip_misses.append(obs_res["candidate_miss_rate"])

    mean_top1 = float(np.mean(trip_top1_accs))
    mean_cont = float(np.mean(trip_conts))
    mean_miss = float(np.mean(trip_misses))

    print(f"Student-Test Top-1 Accuracy: {mean_top1*100:.2f}%")
    print(f"Student-Test Macro F1: {te_metrics['macro_f1']:.4f}")
    print(f"Student-Test Log Loss: {te_metrics['log_loss']:.4f}")
    print(f"Route Continuity (Classifier-only): {mean_cont*100:.2f}%")
    print(f"Mean Inference Time: {avg_infer_per_trip:.2f} ms / trip")

    # Confusion matrix
    cm = confusion_matrix(y_te, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Negative (0)", "Positive (1)"],
                yticklabels=["Negative (0)", "Positive (1)"])
    plt.title(f"Confusion Matrix: {model_display_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"artifacts/plots/models/cm_{key}.png", dpi=300)
    plt.close()

    comparison_results.append({
        "Model": model_display_name,
        "Algorithm Family": family,
        "Key Configuration": desc,
        "CV Macro F1-score": round(mean_cv_macro_f1, 4),
        "Top-1 Accuracy": f"{round(mean_top1 * 100, 2)}%",
        "Route continuity": f"{round(mean_cont * 100, 2)}%",
        "Student-Test Macro F1-score": round(te_metrics["macro_f1"], 4),
        "Log Loss": round(te_metrics["log_loss"], 4) if not np.isnan(te_metrics["log_loss"]) else "N/A",
        "Inference Time": f"{round(avg_infer_per_trip, 1)} ms/trip",
        "Training Time": f"{round(fit_time, 2)} s",
        "Raw_Top1": mean_top1,
        "Raw_CV_F1": mean_cv_macro_f1,
        "Raw_Test_F1": te_metrics["macro_f1"],
        "Raw_Log_Loss": te_metrics["log_loss"] if not np.isnan(te_metrics["log_loss"]) else 999.0,
    })

# Save comparison table
df_comp = pd.DataFrame(comparison_results)
df_comp.to_csv("artifacts/results/model_comparison.csv", index=False)
print("\n=== CONSOLIDATED MODEL COMPARISON TABLE ===")
display_cols = [
    "Model", "Algorithm Family", "Key Configuration", "CV Macro F1-score",
    "Top-1 Accuracy", "Route continuity", "Student-Test Macro F1-score",
    "Log Loss", "Inference Time"
]
print(df_comp[display_cols].to_string(index=False))

# Generate Comparison Visualizations
# 1. CV Macro F1 comparison across all 5 models
plt.figure(figsize=(9, 5))
bars = plt.bar(df_comp["Model"], df_comp["Raw_CV_F1"], color="#1976D2", width=0.55)
plt.ylabel("Grouped CV Macro F1-Score")
plt.title("Cross-Validation Macro F1-Score Comparison Across Algorithm Families")
plt.ylim(0, 1.0)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("artifacts/plots/models/cv_macro_f1_comparison.png", dpi=300)
plt.close()

# 2. Student-Testing Top-1 Accuracy & Macro F1 Comparison
plt.figure(figsize=(10, 5))
x = np.arange(len(df_comp))
width = 0.35
rects1 = plt.bar(x - width/2, df_comp["Raw_Top1"], width, label="Top-1 Accuracy", color="#2E7D32")
rects2 = plt.bar(x + width/2, df_comp["Raw_Test_F1"], width, label="Macro F1-Score", color="#FF8F00")
plt.ylabel("Score")
plt.title("Student-Testing Top-1 Accuracy and Macro F1-Score Comparison")
plt.xticks(x, df_comp["Model"], rotation=20, ha="right")
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
for bar in rects1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval*100:.1f}%", ha="center", va="bottom", fontsize=9)
for bar in rects2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.015, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("artifacts/plots/models/test_top1_f1_comparison.png", dpi=300)
plt.close()

# 3. Log Loss comparison
plt.figure(figsize=(9, 5))
valid_loss_df = df_comp[df_comp["Log Loss"] != "N/A"].copy()
bars = plt.bar(valid_loss_df["Model"], valid_loss_df["Raw_Log_Loss"], color="#C2185B", width=0.5)
plt.ylabel("Cross-Entropy Log Loss (Lower is Better)")
plt.title("Probabilistic Calibration Comparison (Log Loss on Student Test)")
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("artifacts/plots/models/log_loss_comparison.png", dpi=300)
plt.close()

# 4. Training and validation loss curves for HistGradientBoosting (epoch-wise history)
hgb = models_trained["gradient_boosting"]
if hasattr(hgb, "train_score_"):
    plt.figure(figsize=(8, 4.5))
    plt.plot(hgb.train_score_, label="Training Score", color="#1565C0", linewidth=2)
    if hasattr(hgb, "validation_score_") and hgb.validation_score_ is not None and len(hgb.validation_score_) > 0:
        plt.plot(hgb.validation_score_, label="Validation Score", color="#D32F2F", linewidth=2)
    plt.xlabel("Boosting Iteration")
    plt.ylabel("Accuracy / Log-Likelihood Score")
    plt.title("HistGradientBoosting Training History Across Iterations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("artifacts/plots/models/learning_curves.png", dpi=300)
    plt.close()

print("\nModel experimentation and evaluation completed successfully!")

