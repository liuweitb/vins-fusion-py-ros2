"""Feature manager: tracks observations across the sliding window (mirrors feature_manager.h)."""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class FeaturePerId:
    """All observations of a single feature across the window."""

    def __init__(self, feature_id: int, start_frame: int):
        self.feature_id = feature_id
        self.start_frame = start_frame
        self.feature_per_frame: List[np.ndarray] = []  # per-frame observation vector
        self.used_num = 0
        self.estimated_depth: float = -1.0
        self.solve_flag: int = 0  # 0=unset, 1=triangulated, 2=failed


class FeatureManager:
    def __init__(self, window_size: int = 10, min_parallax: float = 10.0):
        self.window_size = window_size
        self.min_parallax = min_parallax
        self.feature: Dict[int, FeaturePerId] = {}

    def clear_state(self):
        self.feature.clear()

    def add_feature_check_parallax(self,
                                    frame_count: int,
                                    image: Dict[int, List[np.ndarray]],
                                    td: float = 0.0
                                    ) -> bool:
        """
        Add new observations.  Returns True if this frame should become a keyframe
        (sufficient parallax with second-newest keyframe).
        """
        parallax_sum = 0.0
        parallax_num = 0
        last_track_num = 0

        for fid, obs_list in image.items():
            obs = obs_list[0]  # camera 0 observation
            if fid not in self.feature:
                self.feature[fid] = FeaturePerId(fid, frame_count)
            feat = self.feature[fid]
            feat.feature_per_frame.append(obs.copy())
            if feat.start_frame < frame_count:
                last_track_num += 1

        if frame_count < 2 or last_track_num < 20:
            return True

        for fid, feat in self.feature.items():
            window_start = frame_count - self.window_size
            if feat.start_frame <= frame_count - 2 and \
               feat.start_frame + len(feat.feature_per_frame) - 1 >= frame_count - 1:
                # Compute parallax between (frame_count-2) and (frame_count-1)
                idx_i = frame_count - 2 - feat.start_frame
                idx_j = frame_count - 1 - feat.start_frame
                if idx_i >= 0 and idx_j < len(feat.feature_per_frame):
                    p_i = feat.feature_per_frame[idx_i][:2]
                    p_j = feat.feature_per_frame[idx_j][:2]
                    parallax_sum += np.linalg.norm(p_i - p_j)
                    parallax_num += 1

        if parallax_num == 0:
            return True
        avg_parallax = parallax_sum / parallax_num * 460.0  # convert to pixels
        return avg_parallax >= self.min_parallax

    def remove_back(self):
        """Slide window: remove oldest frame observations."""
        for fid in list(self.feature.keys()):
            feat = self.feature[fid]
            if feat.start_frame != 0:
                feat.start_frame -= 1
            else:
                feat.feature_per_frame.pop(0)
                if len(feat.feature_per_frame) == 0:
                    del self.feature[fid]

    def remove_front(self, frame_count: int):
        """Slide window: remove second-newest frame."""
        for fid in list(self.feature.keys()):
            feat = self.feature[fid]
            if feat.start_frame == frame_count:
                feat.start_frame -= 1
            else:
                j = frame_count - 1 - feat.start_frame
                if j >= 0 and j < len(feat.feature_per_frame):
                    feat.feature_per_frame.pop(j)
                if len(feat.feature_per_frame) == 0:
                    del self.feature[fid]

    def get_depth_vector(self) -> np.ndarray:
        depths = []
        for feat in self.feature.values():
            if feat.estimated_depth > 0:
                depths.append(feat.estimated_depth)
            else:
                depths.append(-1.0)
        return np.array(depths)

    def set_depth(self, x: np.ndarray):
        i = 0
        for feat in self.feature.values():
            feat.estimated_depth = x[i]
            i += 1
            if i >= len(x):
                break

    def triangulate(self, Ps: List[np.ndarray], Rs: List[np.ndarray],
                    tic: np.ndarray, ric: np.ndarray) -> None:
        """Triangulate all uninitialized features."""
        for feat in self.feature.values():
            if feat.estimated_depth > 0:
                continue
            imu_i = feat.start_frame
            pts_i = feat.feature_per_frame[0][:3]
            pts_i = pts_i / np.linalg.norm(pts_i)

            A = []
            b_vec = []
            for j, obs in enumerate(feat.feature_per_frame):
                imu_j = imu_i + j
                if imu_j >= len(Ps):
                    break
                pts_j = obs[:3] / np.linalg.norm(obs[:3])
                Ri = Rs[imu_i]
                Rj = Rs[imu_j]
                Pi_w = Ps[imu_i]
                Pj_w = Ps[imu_j]
                # In camera frame
                R_ic = ric
                t_ic = tic
                R_ci = R_ic.T
                R_cj = R_ic.T

                p_w_i = Ri @ (R_ic @ pts_i)  # direction in world
                A.append(p_w_i[:2])
                b_vec.append(0.0)

            # Simple linear triangulation
            if len(feat.feature_per_frame) >= 2:
                obs0 = feat.feature_per_frame[0][:3]
                obs1 = feat.feature_per_frame[1][:3]
                imu_j = imu_i + 1
                if imu_j < len(Ps):
                    p3d = self._triangulate_dlt(
                        Rs[imu_i], Ps[imu_i], Rs[imu_j], Ps[imu_j],
                        ric, tic, obs0, obs1)
                    if p3d is not None:
                        p_cam = ric.T @ (Rs[imu_i].T @ (p3d - Ps[imu_i]) - tic)
                        if p_cam[2] > 0.1:
                            feat.estimated_depth = p_cam[2]
                            feat.solve_flag = 1
                        else:
                            feat.solve_flag = 2

    @staticmethod
    def _triangulate_dlt(R0: np.ndarray, P0: np.ndarray,
                          R1: np.ndarray, P1: np.ndarray,
                          Ric: np.ndarray, tic: np.ndarray,
                          obs0: np.ndarray, obs1: np.ndarray
                          ) -> Optional[np.ndarray]:
        """DLT triangulation in world frame."""
        Rc0 = Ric.T @ R0.T
        tc0 = -Ric.T @ (R0.T @ P0 + tic)  # camera 0 position in world? no...
        # Build projection matrices (world -> image)
        t0 = -(Ric.T @ R0.T @ P0 + Ric.T @ tic)  # = -(R_cw * t_w)... fix
        t0 = Ric.T @ (R0.T @ (-P0) - tic)
        t1 = Ric.T @ (R1.T @ (-P1) - tic)
        P_mat0 = np.hstack([Rc0, t0.reshape(3, 1)])
        P_mat1 = np.hstack([Ric.T @ R1.T, t1.reshape(3, 1)])

        A = np.array([
            obs0[0]*P_mat0[2] - P_mat0[0],
            obs0[1]*P_mat0[2] - P_mat0[1],
            obs1[0]*P_mat1[2] - P_mat1[0],
            obs1[1]*P_mat1[2] - P_mat1[1],
        ])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        if abs(X[3]) < 1e-10:
            return None
        return X[:3] / X[3]
