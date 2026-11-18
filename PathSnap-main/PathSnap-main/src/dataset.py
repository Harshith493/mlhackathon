"""
dataset.py - Core data loading, coordinate transformation, spatial indexing,
and candidate generation for the PathSnap Map-Matching system.
"""

import os
from collections import defaultdict
import numpy as np
from scipy.spatial import cKDTree


def load_trip_data(trip_dir):
    """
    Loads raw trip data files: .track, .nodes, .arcs, and optional .route.
    Returns:
        dict with keys: 'trip_id', 'track', 'nodes', 'arcs', 'route' (or None)
    """
    trip_id = os.path.basename(trip_dir.rstrip("/\\"))
    track_path = os.path.join(trip_dir, f"{trip_id}.track")
    nodes_path = os.path.join(trip_dir, f"{trip_id}.nodes")
    arcs_path = os.path.join(trip_dir, f"{trip_id}.arcs")
    route_path = os.path.join(trip_dir, f"{trip_id}.route")

    if not (os.path.exists(track_path) and os.path.exists(nodes_path) and os.path.exists(arcs_path)):
        raise FileNotFoundError(f"Missing essential files in {trip_dir}")

    # Track: lon, lat, timestamp
    with open(track_path, "r", encoding="utf-8") as f:
        track = np.array([[float(x) for x in line.split()[:3]] for line in f if line.strip()], dtype=np.float64)

    # Nodes: lon, lat (line index is node_id)
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = np.array([[float(x) for x in line.split()[:2]] for line in f if line.strip()], dtype=np.float64)

    # Arcs: from_node, to_node (line index is arc_id)
    with open(arcs_path, "r", encoding="utf-8") as f:
        arcs = np.array([[int(x) for x in line.split()[:2]] for line in f if line.strip()], dtype=np.int32)

    # Route: sequence of ground-truth arc_ids (if present)
    route = None
    if os.path.exists(route_path):
        with open(route_path, "r", encoding="utf-8") as f:
            route = [int(line.strip()) for line in f if line.strip()]

    return {
        "trip_id": trip_id,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def project_coordinates(nodes, track, lat_0=None, lon_0=None):
    """
    Converts spherical coordinates (lon, lat in degrees) to local metric planar
    coordinates (x, y in meters) using an equirectangular projection centered
    at (lon_0, lat_0).
    """
    if lat_0 is None:
        lat_0 = float(np.mean(track[:, 1]))
    if lon_0 is None:
        lon_0 = float(np.mean(track[:, 0]))

    kx = 111320.0 * np.cos(np.radians(lat_0))
    ky = 110540.0

    nodes_m = np.empty_like(nodes, dtype=np.float64)
    nodes_m[:, 0] = (nodes[:, 0] - lon_0) * kx
    nodes_m[:, 1] = (nodes[:, 1] - lat_0) * ky

    track_m = np.empty((len(track), 2), dtype=np.float64)
    track_m[:, 0] = (track[:, 0] - lon_0) * kx
    track_m[:, 1] = (track[:, 1] - lat_0) * ky

    meta = {"lat_0": lat_0, "lon_0": lon_0, "kx": kx, "ky": ky}
    return nodes_m, track_m, meta


def point_to_segment(p, a, b):
    """
    Computes Euclidean distance and projection parameter t in [0, 1]
    from point p to line segment ab in 2D metric space.
    """
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    lsq = dx * dx + dy * dy
    if lsq < 1e-9:
        return float(np.hypot(p[0] - a[0], p[1] - a[1])), 0.0
    t = max(0.0, min(1.0, float(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / lsq)))
    proj_x = a[0] + t * dx
    proj_y = a[1] + t * dy
    dist = float(np.hypot(p[0] - proj_x, p[1] - proj_y))
    return dist, t


def align_ground_truth_route(track_m, route, arcs, nodes_m, lookahead=15):
    """
    Aligns each GPS observation to its reference route arc using forward temporal tracking.
    """
    if route is None or len(route) == 0:
        return None, None

    assigned_arcs = []
    assigned_dists = []
    curr_r = 0
    route_len = len(route)

    for i in range(len(track_m)):
        p = track_m[i]
        best_d = float("inf")
        best_r = curr_r
        window_end = min(route_len, curr_r + lookahead)

        for r_idx in range(curr_r, window_end):
            arc_id = route[r_idx]
            u, v = arcs[arc_id]
            a, b = nodes_m[u], nodes_m[v]
            d, _ = point_to_segment(p, a, b)
            if d < best_d:
                best_d = d
                best_r = r_idx

        curr_r = best_r
        assigned_arcs.append(route[best_r])
        assigned_dists.append(best_d)

    return np.array(assigned_arcs, dtype=np.int32), np.array(assigned_dists, dtype=np.float64)


def build_spatial_indices(nodes_m, arcs):
    """
    Builds KD-tree indices on road network nodes and arc midpoints for rapid spatial candidate retrieval.
    """
    u_nodes = arcs[:, 0]
    v_nodes = arcs[:, 1]
    u_pts = nodes_m[u_nodes]
    v_pts = nodes_m[v_nodes]

    # Arc midpoints
    arc_midpoints = (u_pts + v_pts) / 2.0
    mid_tree = cKDTree(arc_midpoints)

    # Node KD-tree
    node_tree = cKDTree(nodes_m)

    # Node to incident arcs mapping
    node_to_arcs = defaultdict(list)
    for arc_id in range(len(arcs)):
        u = arcs[arc_id, 0]
        v = arcs[arc_id, 1]
        node_to_arcs[u].append(arc_id)
        node_to_arcs[v].append(arc_id)

    return {
        "mid_tree": mid_tree,
        "node_tree": node_tree,
        "node_to_arcs": node_to_arcs,
        "u_pts": u_pts,
        "v_pts": v_pts,
    }


def generate_candidates(track_m, arcs, nodes_m, spatial_idx, top_k=10, k_search=15):
    """
    Generates top_k candidate road segments for each GPS observation in track_m.
    Returns:
        candidates: list of length N (one list per GPS observation),
                    where each element is a list of tuples: (dist, proj_t, arc_id)
    """
    mid_tree = spatial_idx["mid_tree"]
    node_tree = spatial_idx["node_tree"]
    node_to_arcs = spatial_idx["node_to_arcs"]
    u_pts = spatial_idx["u_pts"]
    v_pts = spatial_idx["v_pts"]

    _, m_idxs = mid_tree.query(track_m, k=min(k_search, len(arcs)))
    _, n_idxs = node_tree.query(track_m, k=min(k_search, len(nodes_m)))

    candidates = []
    for i in range(len(track_m)):
        p = track_m[i]
        cands_set = set(m_idxs[i])
        for nid in n_idxs[i]:
            cands_set.update(node_to_arcs[nid])

        cand_dists = []
        for cid in cands_set:
            a, b = u_pts[cid], v_pts[cid]
            d, t = point_to_segment(p, a, b)
            cand_dists.append((d, t, cid))

        cand_dists.sort(key=lambda x: x[0])
        candidates.append(cand_dists[:top_k])

    return candidates

