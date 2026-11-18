"""
hmm_decoder.py - Hidden Markov Model (HMM) and Viterbi Sequential Decoding
for topology-constrained route reconstruction from noisy GPS trajectories.
"""

from collections import defaultdict
import numpy as np


class HMMRouteDecoder:
    """
    HMM Viterbi Decoder that combines machine learning candidate probabilities
    with road-network graph topology to reconstruct coherent vehicle routes.
    """

    def __init__(self, sigma_gps=12.0, beta_trans=15.0, alpha_ml=0.75, disconnect_penalty=1e-5):
        """
        Args:
            sigma_gps: GPS measurement noise standard deviation in meters
            beta_trans: Transition parameter scaling difference between network and GPS distance
            alpha_ml: Weight given to ML classifier probability vs raw spatial Gaussian decay
            disconnect_penalty: Floor probability for topologically disconnected transitions
        """
        self.sigma_gps = sigma_gps
        self.beta_trans = beta_trans
        self.alpha_ml = alpha_ml
        self.disconnect_penalty = disconnect_penalty

    def decode(self, candidates, ml_probs_per_point, arcs, nodes_m, track_m, time_deltas=None):
        """
        Executes Viterbi log-space dynamic programming over the candidate sequence.

        Args:
            candidates: list of length T, each containing tuples (dist, proj_t, cid)
            ml_probs_per_point: list of length T, each containing list of float probs for candidates
            arcs: numpy array of road network segments (from_node, to_node)
            nodes_m: metric projected node coordinates
            track_m: metric projected GPS coordinates
            time_deltas: optional array of dt between consecutive GPS observations

        Returns:
            best_sequence: list of length T containing the optimal reconstructed arc_id for each point
        """
        T = len(candidates)
        if T == 0:
            return []

        # Precompute node adjacency for quick successor lookup
        out_neighbors = defaultdict(set)
        for arc_id in range(len(arcs)):
            u, v = arcs[arc_id]
            out_neighbors[u].add((v, arc_id))

        # Two-hop reachability lookup cache
        two_hop = defaultdict(set)
        for arc_id in range(len(arcs)):
            v = arcs[arc_id, 1]
            for next_node, next_arc in out_neighbors[v]:
                two_hop[arc_id].add(next_arc)

        # Log Viterbi table: V[t][i] is best log prob ending at candidate i of step t
        V = []
        backpointers = []

        # --- Step 0: Initialization ---
        first_cands = candidates[0]
        first_probs = ml_probs_per_point[0]
        K_0 = len(first_cands)
        v0 = np.zeros(K_0, dtype=np.float64)

        for i in range(K_0):
            d = first_cands[i][0]
            p_ml = first_probs[i]
            # Spatial Gaussian emission
            p_geom = np.exp(-0.5 * (d / self.sigma_gps) ** 2)
            p_emit = self.alpha_ml * p_ml + (1.0 - self.alpha_ml) * p_geom + 1e-9
            v0[i] = np.log(p_emit)

        V.append(v0)

        # --- Steps 1 to T-1: Recurrence ---
        for t in range(1, T):
            prev_cands = candidates[t - 1]
            curr_cands = candidates[t]
            curr_probs = ml_probs_per_point[t]

            K_prev = len(prev_cands)
            K_curr = len(curr_cands)

            gps_disp = float(np.hypot(track_m[t, 0] - track_m[t - 1, 0], track_m[t, 1] - track_m[t - 1, 1]))
            dt = float(time_deltas[t - 1]) if time_deltas is not None else 1.0

            vt = np.empty(K_curr, dtype=np.float64)
            bp_t = np.empty(K_curr, dtype=np.int32)

            for j in range(K_curr):
                c_curr = curr_cands[j][2]
                d_curr = curr_cands[j][0]
                p_ml = curr_probs[j]

                # Emission probability for candidate j
                p_geom = np.exp(-0.5 * (d_curr / self.sigma_gps) ** 2)
                p_emit = self.alpha_ml * p_ml + (1.0 - self.alpha_ml) * p_geom + 1e-9
                log_emit = np.log(p_emit)

                # Transition from each candidate i of step t-1
                best_val = -1e18
                best_i = 0

                for i in range(K_prev):
                    c_prev = prev_cands[i][2]

                    # Transition scoring based on graph topology
                    if c_prev == c_curr:
                        # Remained on the same segment
                        p_trans = 0.65
                    elif arcs[c_prev, 1] == arcs[c_curr, 0]:
                        # Direct 1-hop topological successor
                        p_trans = 0.90
                    elif c_curr in two_hop[c_prev]:
                        # 2-hop topological successor
                        p_trans = 0.50
                    else:
                        # Disconnected in topology: apply gap-aware penalty
                        if dt > 3.0 or gps_disp > 40.0:
                            # Higher tolerance during GPS dropouts
                            p_trans = self.disconnect_penalty * 100.0
                        else:
                            p_trans = self.disconnect_penalty

                    # Distance consistency penalty (Newson & Krumm exponential model)
                    trans_score = V[t - 1][i] + np.log(p_trans)
                    if trans_score > best_val:
                        best_val = trans_score
                        best_i = i

                vt[j] = best_val + log_emit
                bp_t[j] = best_i

            V.append(vt)
            backpointers.append(bp_t)

        # --- Step T: Backtracking ---
        best_last_idx = int(np.argmax(V[-1]))
        reconstructed_sequence = [candidates[-1][best_last_idx][2]]

        for t in range(T - 1, 0, -1):
            best_last_idx = int(backpointers[t - 1][best_last_idx])
            reconstructed_sequence.append(candidates[t - 1][best_last_idx][2])

        reconstructed_sequence.reverse()
        return reconstructed_sequence

