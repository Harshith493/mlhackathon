"""
run_bayesian_optimization_pipeline.py - Executes Bayesian Hyperparameter Optimization
using Optuna for the winning HistGradientBoosting model.
Compares Untuned vs Tuned model performance on Grouped CV and Student-Testing data.
"""

import glob
import os
import time
import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.ensemble import HistGradientBoostingClassifier
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

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

os.makedirs("artifacts/plots/bayesian_opt", exist_ok=True)
os.makedirs("artifacts/results", exist_ok=True)

train_dirs = sorted(glob.glob("Student folder_extracted/Student folder/train/*"))
test_dirs = sorted(glob.glob("Student folder_extracted/Student folder/student_test/*"))
selected_train_dirs = train_dirs[::3][:25]


def load_dataset(trip_dirs):
    X_all, y_all, groups_all, meta_all = [], [], [], []
    arcs_by_trip = {}
    test_store = {}

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
        test_store[trip_id] = {
            "trip_data": trip_data,
            "nodes_m": nodes_m,
            "track_m": track_m,
            "candidates": candidates,
            "assigned_arcs": assigned_arcs,
        }

    return np.vstack(X_all), np.concatenate(y_all), np.array(groups_all), meta_all, arcs_by_trip, test_store


print("Loading train and test features for Bayesian optimization...")
X_tr, y_tr, groups_tr, meta_tr, arcs_tr, _ = load_dataset(selected_train_dirs)
X_te, y_te, groups_te, meta_te, arcs_te, test_store = load_dataset(test_dirs)

print(f"Train samples: {len(X_tr)}, Test samples: {len(X_te)}")

# Define Grouped CV Objective
gkf = GroupKFold(n_splits=3)


def objective(trial):
    lr = trial.suggest_float("learning_rate", 0.03, 0.25, log=True)
    max_iter = trial.suggest_int("max_iter", 90, 220)
    max_depth = trial.suggest_int("max_depth", 5, 12)
    min_leaf = trial.suggest_int("min_samples_leaf", 15, 60)
    l2 = trial.suggest_float("l2_regularization", 1e-3, 5.0, log=True)

    cv_scores = []
    for f_tr, f_val in gkf.split(X_tr, y_tr, groups=groups_tr):
        clf = HistGradientBoostingClassifier(
            learning_rate=lr,
            max_iter=max_iter,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            l2_regularization=l2,
            random_state=42,
        )
        clf.fit(X_tr[f_tr], y_tr[f_tr])
        val_preds = clf.predict(X_tr[f_val])
        val_probs = clf.predict_proba(X_tr[f_val])[:, 1]
        m = compute_classification_metrics(y_tr[f_val], val_preds, val_probs)
        cv_scores.append(m["macro_f1"])

    return float(np.mean(cv_scores))


print("Launching Optuna study (20 trials)...")
optuna.logging.set_verbosity(optuna.logging.INFO)
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=20)

print(f"\nBest Trial: #{study.best_trial.number}")
print(f"Best CV Macro F1: {study.best_value:.4f}")
print("Best Hyperparameters:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# Save trials dataframe
trials_df = study.trials_dataframe()
trials_df.to_csv("artifacts/results/bayesian_trials.csv", index=False)

# Train Tuned Model with Best Hyperparameters
print("\nFitting final tuned HistGradientBoosting model...")
tuned_model = HistGradientBoostingClassifier(
    learning_rate=study.best_params["learning_rate"],
    max_iter=study.best_params["max_iter"],
    max_depth=study.best_params["max_depth"],
    min_samples_leaf=study.best_params["min_samples_leaf"],
    l2_regularization=study.best_params["l2_regularization"],
    random_state=42,
)

t0 = time.time()
tuned_model.fit(X_tr, y_tr)
tuned_fit_time = time.time() - t0
joblib.dump(tuned_model, "artifacts/models/gradient_boosting_tuned.joblib")

# Evaluate Tuned Model on Student-Test Trips
infer_t0 = time.time()
te_preds_tuned = tuned_model.predict(X_te)
te_probs_tuned = tuned_model.predict_proba(X_te)[:, 1]
tuned_infer_time = ((time.time() - infer_t0) * 1000.0) / len(test_dirs)
te_metrics_tuned = compute_classification_metrics(y_te, te_preds_tuned, te_probs_tuned)

# Observation-level Top-1 Accuracy on Student-Test Trips
trip_top1_tuned = []
for tid in sorted(test_store.keys()):
    t_indices = [k for k, m in enumerate(meta_te) if m[0] == tid]
    t_sub_meta = [(meta_te[k][1], meta_te[k][2]) for k in t_indices]
    t_y = y_te[t_indices]
    t_probs = te_probs_tuned[t_indices]
    t_arcs = arcs_te[tid]
    obs_res = compute_observation_metrics(t_sub_meta, t_y, t_probs, t_arcs)
    trip_top1_tuned.append(obs_res["top1_accuracy"])

top1_tuned = float(np.mean(trip_top1_tuned))

# Load Untuned Baseline Metrics
untuned_model = joblib.load("artifacts/models/gradient_boosting.joblib")
infer_u0 = time.time()
te_preds_untuned = untuned_model.predict(X_te)
te_probs_untuned = untuned_model.predict_proba(X_te)[:, 1]
untuned_infer_time = ((time.time() - infer_u0) * 1000.0) / len(test_dirs)
te_metrics_untuned = compute_classification_metrics(y_te, te_preds_untuned, te_probs_untuned)

trip_top1_untuned = []
for tid in sorted(test_store.keys()):
    t_indices = [k for k, m in enumerate(meta_te) if m[0] == tid]
    t_sub_meta = [(meta_te[k][1], meta_te[k][2]) for k in t_indices]
    t_y = y_te[t_indices]
    t_probs = te_probs_untuned[t_indices]
    t_arcs = arcs_te[tid]
    obs_res = compute_observation_metrics(t_sub_meta, t_y, t_probs, t_arcs)
    trip_top1_untuned.append(obs_res["top1_accuracy"])
top1_untuned = float(np.mean(trip_top1_untuned))

# Build Comparative Table
comparison_rows = [
    {
        "Model Variant": "Untuned HistGradientBoosting (Default)",
        "CV Macro F1-score": 0.8849,
        "Student-testing Top-1 Accuracy": f"{top1_untuned * 100:.2f}%",
        "Student-testing Macro F1-score": round(te_metrics_untuned["macro_f1"], 4),
        "Log Loss": round(te_metrics_untuned["log_loss"], 4),
        "Inference Time": f"{untuned_infer_time:.1f} ms/trip",
    },
    {
        "Model Variant": "Tuned HistGradientBoosting (Bayesian TPE)",
        "CV Macro F1-score": round(study.best_value, 4),
        "Student-testing Top-1 Accuracy": f"{top1_tuned * 100:.2f}%",
        "Student-testing Macro F1-score": round(te_metrics_tuned["macro_f1"], 4),
        "Log Loss": round(te_metrics_tuned["log_loss"], 4),
        "Inference Time": f"{tuned_infer_time:.1f} ms/trip",
    },
]

df_tune_comp = pd.DataFrame(comparison_rows)
df_tune_comp.to_csv("artifacts/results/tuned_vs_untuned_comparison.csv", index=False)
print("\n=== TUNED VS UNTUNED MODEL COMPARISON ===")
print(df_tune_comp.to_string(index=False))

# Visualization 1: Optimization History
trial_vals = [t.value for t in study.trials if t.value is not None]
best_vals = np.maximum.accumulate(trial_vals)

plt.figure(figsize=(9, 4.5))
plt.plot(range(1, len(trial_vals) + 1), trial_vals, marker="o", linestyle="", color="#1E88E5", alpha=0.7, label="Trial Objective Value")
plt.plot(range(1, len(best_vals) + 1), best_vals, linestyle="-", color="#D32F2F", linewidth=2.5, label="Best Objective So Far")
plt.xlabel("Bayesian Optimization Trial")
plt.ylabel("Grouped CV Macro F1-Score")
plt.title("Bayesian Optimization History (HistGradientBoosting via TPE)")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/plots/bayesian_opt/opt_history.png", dpi=300)
plt.close()

# Visualization 2: Parameter Importance
try:
    importances = optuna.importance.get_param_importances(study)
    plt.figure(figsize=(8, 4))
    param_names = list(importances.keys())
    param_scores = list(importances.values())
    plt.barh(param_names[::-1], param_scores[::-1], color="#00897B")
    plt.xlabel("Relative Importance")
    plt.title("Hyperparameter Importance in Bayesian Optimization")
    plt.tight_layout()
    plt.savefig("artifacts/plots/bayesian_opt/param_importances.png", dpi=300)
    plt.close()
except Exception as e:
    print(f"Note: Param importance plot skipped ({e})")

print("\nBayesian hyperparameter optimization completed successfully!")

