"""
Pose graph with loop-closure detection and 4-DOF optimisation
(mirrors pose_graph.h/cpp).
"""
import threading
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
from typing import Dict, List, Optional, Tuple

import cv2

from .keyframe import KeyFrame


def _yaw_from_R(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _R_from_yaw(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


class PoseGraphEdge:
    def __init__(self, i: int, j: int,
                 relative_t: np.ndarray,
                 relative_yaw: float,
                 is_loop: bool = False):
        self.i = i
        self.j = j
        self.relative_t = relative_t.copy()
        self.relative_yaw = relative_yaw
        self.is_loop = is_loop


class PoseGraph:
    """
    Maintains a graph of keyframe poses, detects loop closures via
    ORB descriptor matching, and runs 4-DOF pose-graph optimisation.
    """

    def __init__(self, min_loop_inliers: int = 15):
        self.keyframes: List[KeyFrame] = []
        self.edges: List[PoseGraphEdge] = []
        self.min_loop_inliers = min_loop_inliers
        self._lock = threading.Lock()

        # ORB matcher for bag-of-words-style retrieval
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Drift-corrected poses (updated after each optimisation)
        self.drift_t = np.zeros(3)
        self.drift_yaw = 0.0

        self.sequence = 0

    # ------------------------------------------------------------------ #
    #  Adding keyframes                                                    #
    # ------------------------------------------------------------------ #
    def add_keyframe(self, kf: KeyFrame, skip_loop: bool = False) -> Optional[int]:
        """
        Add a keyframe to the graph. Attempt loop detection unless skip_loop.
        Returns matched frame index if a loop is found, else None.
        """
        with self._lock:
            idx = len(self.keyframes)
            self.keyframes.append(kf)

            # Sequential edge to previous keyframe
            if idx > 0:
                prev = self.keyframes[idx - 1]
                rel_t = kf.R_w_i.T @ (kf.T_w_i - prev.T_w_i)
                rel_yaw = _yaw_from_R(prev.R_w_i.T @ kf.R_w_i)
                self.edges.append(PoseGraphEdge(idx - 1, idx, rel_t, rel_yaw))

            if skip_loop or idx < 20:
                return None

            loop_idx = self._detect_loop(idx)
            if loop_idx is None:
                return None

            kf_loop = self.keyframes[loop_idx]
            ok, rel_t, rel_R = kf.find_connection(kf_loop, self.min_loop_inliers)
            if not ok:
                return None

            rel_yaw = _yaw_from_R(rel_R)
            self.edges.append(PoseGraphEdge(loop_idx, idx, rel_t, rel_yaw, is_loop=True))
            kf.has_loop = True
            kf.loop_index = loop_idx
            kf.relative_t = rel_t.copy()
            kf.relative_yaw = rel_yaw

            self._optimise_4dof()
            return loop_idx

    # ------------------------------------------------------------------ #
    #  Loop detection                                                      #
    # ------------------------------------------------------------------ #
    def _detect_loop(self, query_idx: int) -> Optional[int]:
        """
        Simple descriptor-bag loop detector: score each earlier frame by
        the number of ORB matches above a threshold.

        Returns the best candidate index or None.
        """
        kf_q = self.keyframes[query_idx]
        if kf_q.descriptors is None or len(kf_q.descriptors) == 0:
            return None

        best_idx = None
        best_score = 30   # minimum matches to consider a loop

        # Only search frames far enough in the past (skip last 50)
        search_end = max(0, query_idx - 50)
        for i in range(search_end):
            kf_i = self.keyframes[i]
            if kf_i.descriptors is None or len(kf_i.descriptors) == 0:
                continue
            matches = self._matcher.match(kf_q.descriptors, kf_i.descriptors)
            score = len(matches)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    # ------------------------------------------------------------------ #
    #  4-DOF pose graph optimisation                                       #
    # ------------------------------------------------------------------ #
    def _optimise_4dof(self):
        """
        Optimise poses (x, y, z, yaw) subject to sequential and loop edges.
        Roll and pitch are fixed because IMU constrains them.
        """
        n = len(self.keyframes)
        if n < 2:
            return

        # Pack: [x, y, z, yaw] per keyframe = 4n parameters
        x0 = np.zeros(4 * n)
        for i, kf in enumerate(self.keyframes):
            x0[4*i:4*i+3] = kf.T_w_i
            x0[4*i+3] = _yaw_from_R(kf.R_w_i)

        def cost(x):
            total = 0.0
            for e in self.edges:
                i, j = e.i, e.j
                pi = x[4*i:4*i+3]
                yi = x[4*i+3]
                pj = x[4*j:4*j+3]
                yj = x[4*j+3]
                Ri = _R_from_yaw(yi)
                rel_t_pred = Ri.T @ (pj - pi)
                dt = rel_t_pred - e.relative_t
                dyaw = (yj - yi - e.relative_yaw + np.pi) % (2*np.pi) - np.pi
                w = 10.0 if e.is_loop else 1.0
                total += w * (np.dot(dt, dt) + dyaw**2)
            return total

        # Fix first frame
        bounds = [(None, None)] * (4 * n)
        bounds[0] = (x0[0], x0[0])
        bounds[1] = (x0[1], x0[1])
        bounds[2] = (x0[2], x0[2])
        bounds[3] = (x0[3], x0[3])

        result = minimize(cost, x0, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 200, "ftol": 1e-6})
        xopt = result.x

        for i, kf in enumerate(self.keyframes):
            kf.T_w_i = xopt[4*i:4*i+3].copy()
            yaw = xopt[4*i+3]
            # Reconstruct full rotation: keep original roll/pitch, update yaw
            orig_yaw = _yaw_from_R(kf.R_w_i)
            delta_yaw = yaw - orig_yaw
            R_correction = _R_from_yaw(delta_yaw)
            kf.R_w_i = R_correction @ kf.R_w_i

    # ------------------------------------------------------------------ #
    #  Query corrected pose                                                #
    # ------------------------------------------------------------------ #
    def get_pose(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position, R_w_i) for keyframe idx."""
        kf = self.keyframes[idx]
        return kf.T_w_i.copy(), kf.R_w_i.copy()
