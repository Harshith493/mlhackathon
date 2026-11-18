"""
features.py - Feature engineering for candidate road segment scoring.
Computes 12 geometric, kinematic, and topological features for each (GPS observation, candidate arc) pair.
"""

import numpy as np

FEATURE_NAMES = [
    "perp_dist",
    "proj_t",
    "dist_to_u",
    "dist_to_v",
    "arc_length",
    "heading_diff",
    "dist_rank",
    "delta_dist",
    "speed",
    "speed_length_ratio",
    "u_degree",
    "v_degree",
]


def extract_features_for_trip(trip_data, nodes_m, track_m, candidates, assigned_arcs=None):
    """
    Extracts feature matrix X and target labels y (if assigned_arcs provided) for a trip.

    Args:
        trip_data: dict from load_trip_data
        nodes_m: metric projected node coordinates
        track_m: metric projected track coordinates
        candidates: list of top candidates per point from generate_candidates
        assigned_arcs: optional ground truth aligned arcs for training/evaluation

    Returns:
        X: numpy array of shape (N_samples, 12)
        y: numpy array of shape (N_samples,) or None
        meta: list of tuples (point_idx, candidate_arc_id)
    """
    track = trip_data["track"]
    arcs = trip_data["arcs"]
    nodes = trip_data["nodes"]
    num_nodes = len(nodes)

    u_nodes = arcs[:, 0]
    v_nodes = arcs[:, 1]
    u_pts = nodes_m[u_nodes]
    v_pts = nodes_m[v_nodes]

    # Arc lengths and orientations
    arc_vecs = v_pts - u_pts
    arc_lens = np.hypot(arc_vecs[:, 0], arc_vecs[:, 1])
    arc_headings = np.degrees(np.arctan2(arc_vecs[:, 1], arc_vecs[:, 0])) % 360.0

    # Node degrees
    u_degs = np.bincount(u_nodes, minlength=num_nodes)
    v_degs = np.bincount(v_nodes, minlength=num_nodes)

    # GPS kinematics (speed and heading)
    if len(track_m) > 1:
        diff_pts = np.diff(track_m, axis=0)
        dt = np.diff(track[:, 2])
        dt[dt <= 0] = 1.0  # avoid division by zero
        speeds = np.hypot(diff_pts[:, 0], diff_pts[:, 1]) / dt
        speeds = np.concatenate([[speeds[0]], speeds])

        headings = np.degrees(np.arctan2(diff_pts[:, 1], diff_pts[:, 0])) % 360.0
        headings = np.concatenate([[headings[0]], headings])
    else:
        speeds = np.zeros(len(track_m))
        headings = np.zeros(len(track_m))

    X_list = []
    y_list = [] if assigned_arcs is not None else None
    meta_list = []

    for i in range(len(track_m)):
        p = track_m[i]
        cands = candidates[i]
        if not cands:
            continue

        min_d = cands[0][0]
        true_arc = assigned_arcs[i] if assigned_arcs is not None else -1

        for rank, (d, t, cid) in enumerate(cands):
            u_node = u_nodes[cid]
            v_node = v_nodes[cid]

            d_u = float(np.hypot(p[0] - u_pts[cid, 0], p[1] - u_pts[cid, 1]))
            d_v = float(np.hypot(p[0] - v_pts[cid, 0], p[1] - v_pts[cid, 1]))

            # Heading difference in [0, 180]
            h_diff = abs(headings[i] - arc_headings[cid])
            if h_diff > 180.0:
                h_diff = 360.0 - h_diff

            feat = [
                d,
                t,
                d_u,
                d_v,
                arc_lens[cid],
                h_diff,
                float(rank),
                d - min_d,
                speeds[i],
                speeds[i] / (arc_lens[cid] + 1e-3),
                float(u_degs[u_node]),
                float(v_degs[v_node]),
            ]

            X_list.append(feat)
            meta_list.append((i, cid))
            if y_list is not None:
                y_list.append(1 if cid == true_arc else 0)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list is not None else None
    return X, y, meta_list

