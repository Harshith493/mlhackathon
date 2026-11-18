"""
bayesian_opt.py - Optuna-based Bayesian Hyperparameter Optimization
for the winning classifier using Group-Aware Cross-Validation.
"""

import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from src.metrics import compute_observation_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


def run_bayesian_optimization(X_train, y_train, groups_train, meta_train, arcs_by_trip, n_trials=25):
    """
    Executes Bayesian hyperparameter search using Tree-structured Parzen Estimator (TPE).

    Args:
        X_train: numpy array of features
        y_train: numpy array of binary labels
        groups_train: array of trip_ids identifying which trip each sample belongs to
        meta_train: list of (trip_id, point_idx, cid)
        arcs_by_trip: dict mapping trip_id to arcs array
        n_trials: number of optimization iterations

    Returns:
        study: Optuna Study object
        best_params: dict of optimal hyperparameter values
    """
    gkf = GroupKFold(n_splits=3)

    def objective(trial):
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.25, log=True)
        max_iter = trial.suggest_int("max_iter", 80, 220)
        max_depth = trial.suggest_int("max_depth", 4, 14)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 10, 60)
        l2_reg = trial.suggest_float("l2_regularization", 1e-3, 5.0, log=True)

        fold_accs = []
        for train_idx, val_idx in gkf.split(X_train, y_train, groups=groups_train):
            clf = HistGradientBoostingClassifier(
                learning_rate=learning_rate,
                max_iter=max_iter,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                l2_regularization=l2_reg,
                random_state=42,
            )
            clf.fit(X_train[train_idx], y_train[train_idx])
            val_probs = clf.predict_proba(X_train[val_idx])[:, 1]

            # Compute observation-level top1 accuracy for validation fold
            val_meta = [meta_train[i] for i in val_idx]
            val_y = y_train[val_idx]

            # Group by trip
            val_trips = set(m[0] for m in val_meta)
            trip_accs = []
            for tid in val_trips:
                t_indices = [k for k, m in enumerate(val_meta) if m[0] == tid]
                t_sub_meta = [(val_meta[k][1], val_meta[k][2]) for k in t_indices]
                t_y = val_y[t_indices]
                t_probs = val_probs[t_indices]
                t_arcs = arcs_by_trip[tid]

                res = compute_observation_metrics(t_sub_meta, t_y, t_probs, t_arcs)
                trip_accs.append(res["top1_accuracy"])

            fold_accs.append(sum(trip_accs) / max(1, len(trip_accs)))

        return float(sum(fold_accs) / len(fold_accs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)

    # Save trial results
    trials_df = study.trials_dataframe()
    trials_df.to_csv("artifacts/results/bayesian_trials.csv", index=False)

    return study, study.best_params

