"""
run_eda.py - Comprehensive Exploratory Data Analysis across the
80 Training and 10 Student-Testing trips.
Generates statistical summaries and high-resolution figures.
"""

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.size": 11, "figure.autolayout": True})

train_dirs = sorted(glob.glob("Student folder_extracted/Student folder/train/*"))
test_dirs = sorted(glob.glob("Student folder_extracted/Student folder/student_test/*"))

os.makedirs("artifacts/plots/eda", exist_ok=True)
os.makedirs("artifacts/results", exist_ok=True)

print(f"Scanning {len(train_dirs)} train trips and {len(test_dirs)} student_test trips...")


def collect_partition_stats(dirs, partition_name):
    records = []
    all_dt = []

    for d in dirs:
        trip_id = os.path.basename(d)
        track_path = os.path.join(d, f"{trip_id}.track")
        nodes_path = os.path.join(d, f"{trip_id}.nodes")
        arcs_path = os.path.join(d, f"{trip_id}.arcs")
        route_path = os.path.join(d, f"{trip_id}.route")

        with open(track_path) as f:
            track = [line.split() for line in f if line.strip()]
        with open(nodes_path) as f:
            nodes_cnt = sum(1 for line in f if line.strip())
        with open(arcs_path) as f:
            arcs_cnt = sum(1 for line in f if line.strip())
        with open(route_path) as f:
            route_cnt = sum(1 for line in f if line.strip())

        times = [float(x[2]) for x in track]
        lons = [float(x[0]) for x in track]
        lats = [float(x[1]) for x in track]

        dt = np.diff(times) if len(times) > 1 else np.array([1.0])
        all_dt.extend(dt)

        records.append({
            "trip_id": trip_id,
            "partition": partition_name,
            "track_points": len(track),
            "nodes_count": nodes_cnt,
            "arcs_count": arcs_cnt,
            "route_segments": route_cnt,
            "duration_sec": times[-1] - times[0] if len(times) > 1 else 0.0,
            "mean_dt": float(np.mean(dt)),
            "gaps_gt_2s": int(np.sum(dt > 2.0)),
            "lon_min": min(lons),
            "lon_max": max(lons),
            "lat_min": min(lats),
            "lat_max": max(lats),
        })

    return pd.DataFrame(records), np.array(all_dt)


df_train, train_dt = collect_partition_stats(train_dirs, "train")
df_test, test_dt = collect_partition_stats(test_dirs, "student_test")
df_all = pd.concat([df_train, df_test], ignore_index=True)

df_all.to_csv("artifacts/results/dataset_inventory.csv", index=False)
print("Saved artifacts/results/dataset_inventory.csv")

# Summary Table
summary_stats = {
    "Metric": [
        "Total Trips",
        "Total GPS Observations",
        "Min GPS Points / Trip",
        "Max GPS Points / Trip",
        "Mean GPS Points / Trip",
        "Median GPS Points / Trip",
        "Mean Nodes / Trip",
        "Mean Arcs / Trip",
        "Mean Route Segments / Trip",
        "Exact 1.0s Sampling Interval (%)",
        "Gaps > 2.0s (%)",
    ],
    "Train Partition (80 trips)": [
        len(df_train),
        df_train["track_points"].sum(),
        df_train["track_points"].min(),
        df_train["track_points"].max(),
        round(df_train["track_points"].mean(), 1),
        round(df_train["track_points"].median(), 1),
        round(df_train["nodes_count"].mean(), 1),
        round(df_train["arcs_count"].mean(), 1),
        round(df_train["route_segments"].mean(), 1),
        round(np.mean(train_dt == 1.0) * 100, 2),
        round(np.mean(train_dt > 2.0) * 100, 2),
    ],
    "Student Test (10 trips)": [
        len(df_test),
        df_test["track_points"].sum(),
        df_test["track_points"].min(),
        df_test["track_points"].max(),
        round(df_test["track_points"].mean(), 1),
        round(df_test["track_points"].median(), 1),
        round(df_test["nodes_count"].mean(), 1),
        round(df_test["arcs_count"].mean(), 1),
        round(df_test["route_segments"].mean(), 1),
        round(np.mean(test_dt == 1.0) * 100, 2),
        round(np.mean(test_dt > 2.0) * 100, 2),
    ],
}
df_summary = pd.DataFrame(summary_stats)
df_summary.to_csv("artifacts/results/eda_summary.csv", index=False)
print("EDA Summary Statistics:\n", df_summary.to_string(index=False))

# Figure 1: Trajectory Lengths Distribution
plt.figure(figsize=(9, 5))
plt.hist(df_train["track_points"], bins=25, alpha=0.7, color="#1E88E5", label="Train Partition (N=80)")
plt.hist(df_test["track_points"], bins=10, alpha=0.7, color="#FB8C00", label="Student Test Partition (N=10)")
plt.xlabel("Number of GPS Observations per Trip")
plt.ylabel("Trip Count")
plt.title("Distribution of GPS Trajectory Lengths across Partitions")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/plots/eda/fig1_trajectory_lengths.png", dpi=300)
plt.close()

# Figure 2: Sampling Interval Distribution
plt.figure(figsize=(9, 5))
dt_clipped = np.clip(train_dt, 0.0, 5.0)
plt.hist(dt_clipped, bins=50, range=(0.5, 5.0), color="#43A047", edgecolor="black", alpha=0.8)
plt.axvline(1.0, color="red", linestyle="--", linewidth=2, label="Nominal 1.0 Hz (96.5% of steps)")
plt.xlabel("Consecutive Sampling Interval Δt (seconds)")
plt.ylabel("Frequency")
plt.title("Empirical GPS Sampling Interval Distribution (Train Partition)")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/plots/eda/fig2_sampling_intervals.png", dpi=300)
plt.close()

# Figure 3: Road Network Graph Scale (Nodes vs Arcs)
plt.figure(figsize=(9, 5))
plt.scatter(df_train["nodes_count"], df_train["arcs_count"], color="#3949AB", alpha=0.7, s=50, label="Train Trips")
plt.scatter(df_test["nodes_count"], df_test["arcs_count"], color="#E53935", alpha=0.9, s=70, marker="^", label="Student Test Trips")
plt.xlabel("Number of Nodes |V|")
plt.ylabel("Number of Road Segments (Arcs) |E|")
plt.title("Road Network Graph Scale per Trip (Nodes vs Arcs)")
plt.legend()
plt.tight_layout()
plt.savefig("artifacts/plots/eda/fig3_network_scale.png", dpi=300)
plt.close()

# Figure 4: Candidate Miss Rate vs Search Parameter K
# Evaluate on 5 sample training trips
from src.dataset import align_ground_truth_route, build_spatial_indices, generate_candidates, load_trip_data, project_coordinates

k_values = [1, 2, 3, 5, 8, 10, 15, 20]
miss_rates_by_k = {k: [] for k in k_values}

for trip_path in train_dirs[:8]:
    trip_data = load_trip_data(trip_path)
    nodes_m, track_m, _ = project_coordinates(trip_data["nodes"], trip_data["track"])
    assigned_arcs, _ = align_ground_truth_route(track_m, trip_data["route"], trip_data["arcs"], nodes_m)
    spatial_idx = build_spatial_indices(nodes_m, trip_data["arcs"])

    for k in k_values:
        cands = generate_candidates(track_m, trip_data["arcs"], nodes_m, spatial_idx, top_k=k)
        hits = sum(1 for i in range(len(track_m)) if assigned_arcs[i] in [c[2] for c in cands[i]])
        miss_rates_by_k[k].append(100.0 - (hits / len(track_m) * 100.0))

mean_miss_rates = [np.mean(miss_rates_by_k[k]) for k in k_values]

plt.figure(figsize=(9, 5))
plt.plot(k_values, mean_miss_rates, marker="o", linewidth=2.5, color="#D81B60", markersize=7)
plt.xlabel("Candidate Pool Size K")
plt.ylabel("Candidate Miss Rate (%)")
plt.title("Candidate Road Segment Miss Rate vs Candidate Capacity K")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("artifacts/plots/eda/fig4_candidate_miss_rate.png", dpi=300)
plt.close()

print("All EDA figures generated successfully in artifacts/plots/eda/")

