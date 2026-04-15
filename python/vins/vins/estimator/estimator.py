"""
Sliding-window VIO estimator (mirrors estimator.h/cpp).

State per frame: P(3), Q(4), V(3), Ba(3), Bg(3)  -> 16 per frame
Extrinsic: tic(3), ric(4) per camera
Feature depths: one inverse-depth per landmark
"""
import threading
import numpy as np
from scipy.optimize import least_squares
from typing import Dict, List, Optional, Tuple
from enum import Enum

from .parameters import Parameters
from .feature_manager import FeatureManager
from ..factor.integration_base import IntegrationBase
from ..factor.projection_factors import project_mono, project_stereo_one_frame
from ..factor.marginalization_factor import MarginalizationInfo
from ..initial.solve_5pts import solve_relative_pose
from ..initial.initial_sfm import GlobalSFM, SFMFeature
from ..initial.initial_alignment import solve_gyro_bias, linear_alignment
from ..utility import (quat_to_rot, rot_to_quat, quat_mul, quat_inv,
                        normalize_quat, skew, small_angle_quat)

G_NORM = 9.81007


class SolverFlag(Enum):
    INITIAL = 0
    NON_LINEAR = 1
    FAILURE = 2


class MarginalizationFlag(Enum):
    MARGIN_OLD = 0
    MARGIN_SECOND_NEW = 1


class ImageFrame:
    def __init__(self, image: dict, t: float):
        self.image = image
        self.t = t
        self.R: np.ndarray = np.eye(3)
        self.T: np.ndarray = np.zeros(3)
        self.pre_integration: Optional[IntegrationBase] = None
        self.is_key_frame: bool = False


class Estimator:
    WINDOW_SIZE = 10

    def __init__(self, params: Parameters):
        self.params = params
        self.feature_manager = FeatureManager(
            params.window_size, params.min_parallax)
        self._lock = threading.Lock()

        self.solver_flag = SolverFlag.INITIAL
        self.margin_flag = MarginalizationFlag.MARGIN_OLD

        self._init_vars()

    # ------------------------------------------------------------------ #
    #  State initialisation                                                #
    # ------------------------------------------------------------------ #
    def _init_vars(self):
        self.frame_count = 0
        self.input_image_cnt = 0

        N = self.WINDOW_SIZE + 1
        self.Ps: List[np.ndarray] = [np.zeros(3) for _ in range(N)]
        self.Vs: List[np.ndarray] = [np.zeros(3) for _ in range(N)]
        self.Rs: List[np.ndarray] = [np.eye(3) for _ in range(N)]
        self.Bas: List[np.ndarray] = [np.zeros(3) for _ in range(N)]
        self.Bgs: List[np.ndarray] = [np.zeros(3) for _ in range(N)]
        self.pre_integrations: List[Optional[IntegrationBase]] = [None] * N

        # Camera-IMU extrinsics
        p = self.params
        self.tic: List[np.ndarray] = []
        self.ric: List[np.ndarray] = []
        for i in range(p.num_of_cam):
            T_body_cam = p.body_T_cam[i] if i < len(p.body_T_cam) else np.eye(4)
            self.ric.append(T_body_cam[:3, :3].copy())
            self.tic.append(T_body_cam[:3, 3].copy())

        self.gravity = np.array([0.0, 0.0, -G_NORM])

        # IMU buffers
        self.acc_0 = np.zeros(3)
        self.gyr_0 = np.zeros(3)
        self.first_imu = True

        # Marginalization prior
        self.last_marginalization_info: Optional[MarginalizationInfo] = None

        # All image frames (for initialisation)
        self.all_image_frame: Dict[float, ImageFrame] = {}
        self.tmp_pre_integration: Optional[IntegrationBase] = None

        # Initialisation state
        self.initial_timestamp = 0.0
        self.open_ex_estimation = False

        self.feature_manager.clear_state()

    def clear_state(self):
        self._init_vars()
        self.solver_flag = SolverFlag.INITIAL

    # ------------------------------------------------------------------ #
    #  IMU processing                                                      #
    # ------------------------------------------------------------------ #
    def process_imu(self, dt: float, acc: np.ndarray, gyr: np.ndarray):
        if self.first_imu:
            self.first_imu = False
            self.acc_0 = acc.copy()
            self.gyr_0 = gyr.copy()

        if self.pre_integrations[self.frame_count] is None:
            self.pre_integrations[self.frame_count] = IntegrationBase(
                self.acc_0, self.gyr_0,
                self.Bas[self.frame_count], self.Bgs[self.frame_count],
                self.params.acc_n, self.params.gyr_n,
                self.params.acc_w, self.params.gyr_w,
            )

        if self.frame_count != 0:
            self.pre_integrations[self.frame_count].push_back(dt, acc, gyr)
            if self.tmp_pre_integration is not None:
                self.tmp_pre_integration.push_back(dt, acc, gyr)

            # Mid-point IMU propagation for state prediction
            gyr_mid = 0.5 * (self.gyr_0 + gyr) - self.Bgs[self.frame_count]
            dq = small_angle_quat(gyr_mid * dt)
            self.Rs[self.frame_count] = self.Rs[self.frame_count] @ quat_to_rot(dq)
            acc_0_ub = self.acc_0 - self.Bas[self.frame_count]
            acc_1_ub = acc - self.Bas[self.frame_count]
            acc_mid = 0.5 * (self.Rs[self.frame_count] @ acc_0_ub +
                             self.Rs[self.frame_count] @ acc_1_ub)
            self.Vs[self.frame_count] += (acc_mid + self.gravity) * dt
            self.Ps[self.frame_count] += (self.Vs[self.frame_count] * dt +
                                           0.5 * (acc_mid + self.gravity) * dt**2)

        self.acc_0 = acc.copy()
        self.gyr_0 = gyr.copy()

    # ------------------------------------------------------------------ #
    #  Image processing                                                    #
    # ------------------------------------------------------------------ #
    def process_image(self,
                       image: Dict[int, List[np.ndarray]],
                       header: float
                       ) -> Optional[Dict]:
        """
        Process a new image frame.

        image: feature_id -> list of observations per camera.
        Returns odometry dict if available.
        """
        is_keyframe = self.feature_manager.add_feature_check_parallax(
            self.frame_count, image, self.params.td)

        if is_keyframe:
            self.margin_flag = MarginalizationFlag.MARGIN_OLD
        else:
            self.margin_flag = MarginalizationFlag.MARGIN_SECOND_NEW

        # Store image frame
        img_frame = ImageFrame(image, header)
        img_frame.pre_integration = self.tmp_pre_integration
        self.all_image_frame[header] = img_frame
        self.tmp_pre_integration = IntegrationBase(
            self.acc_0, self.gyr_0,
            self.Bas[self.frame_count], self.Bgs[self.frame_count],
            self.params.acc_n, self.params.gyr_n,
            self.params.acc_w, self.params.gyr_w,
        )

        if self.solver_flag == SolverFlag.INITIAL:
            if self.params.use_imu:
                if self.frame_count == self.WINDOW_SIZE:
                    ok = self._initial_structure()
                    if ok and (header - self.initial_timestamp) > 0.1:
                        self._solve_odometry()
                        self._slide_window()
                        self.feature_manager.remove_back()
                        self.solver_flag = SolverFlag.NON_LINEAR
                    else:
                        self._slide_window()
                        self.feature_manager.remove_back()
                else:
                    self.frame_count += 1
            else:
                # No IMU: just use feature tracking
                self.solver_flag = SolverFlag.NON_LINEAR
                self._solve_odometry()
                self._slide_window()
                self.feature_manager.remove_back()
        else:
            self._solve_odometry()
            if self._failure_detection():
                self.clear_state()
                return None
            self._slide_window()
            if self.margin_flag == MarginalizationFlag.MARGIN_OLD:
                self.feature_manager.remove_back()
            else:
                self.feature_manager.remove_front(self.frame_count)

        return self._get_odometry(header)

    # ------------------------------------------------------------------ #
    #  Initialisation                                                      #
    # ------------------------------------------------------------------ #
    def _initial_structure(self) -> bool:
        # Estimate gyro bias
        bg = solve_gyro_bias(self.all_image_frame)
        for i in range(self.WINDOW_SIZE + 1):
            self.Bgs[i] = bg
        for frame in self.all_image_frame.values():
            if frame.pre_integration:
                frame.pre_integration.repropagate(np.zeros(3), bg)

        # Find a pair of frames with enough parallax
        l, relative_R, relative_T = self._relative_pose()
        if l < 0:
            return False

        sfm_f: List[SFMFeature] = []
        for fid, feat in self.feature_manager.feature.items():
            sf = SFMFeature()
            sf.id = fid
            for j, obs in enumerate(feat.feature_per_frame):
                frame_idx = feat.start_frame + j
                sf.observation.append((frame_idx, obs[:3].copy()))
            sfm_f.append(sf)

        q_arr = [np.array([1.0, 0.0, 0.0, 0.0])] * (self.WINDOW_SIZE + 1)
        t_arr = [np.zeros(3)] * (self.WINDOW_SIZE + 1)
        sfm = GlobalSFM()
        ok = sfm.construct(self.WINDOW_SIZE + 1, q_arr, t_arr,
                           l, relative_R, relative_T, sfm_f)
        if not ok:
            self.margin_flag = MarginalizationFlag.MARGIN_OLD
            return False

        # Align SfM with all_image_frame
        keys = sorted(self.all_image_frame.keys())
        for i, k in enumerate(keys):
            if i < len(q_arr):
                self.all_image_frame[k].R = quat_to_rot(q_arr[i])
                self.all_image_frame[k].T = t_arr[i]

        # Visual-inertial alignment
        g_est, s, _ = linear_alignment(self.all_image_frame, G_NORM)
        if abs(np.linalg.norm(g_est) - G_NORM) > 1.0 or abs(s) < 0.1:
            return False

        self.gravity = g_est
        self.initial_timestamp = keys[-1]

        # Set initial states from SfM
        for i in range(self.WINDOW_SIZE + 1):
            if i < len(q_arr) and q_arr[i] is not None:
                self.Rs[i] = quat_to_rot(q_arr[i])
                self.Ps[i] = t_arr[i] * s if t_arr[i] is not None else np.zeros(3)

        # Triangulate features
        self.feature_manager.triangulate(self.Ps, self.Rs, self.tic[0], self.ric[0])
        return True

    def _relative_pose(self) -> Tuple[int, np.ndarray, np.ndarray]:
        """Find a reference frame with enough parallax for SfM initialisation."""
        for i in range(self.WINDOW_SIZE, -1, -1):
            pts0 = []
            pts1 = []
            for feat in self.feature_manager.feature.values():
                if feat.start_frame <= i and \
                   feat.start_frame + len(feat.feature_per_frame) - 1 >= self.WINDOW_SIZE:
                    idx_i = i - feat.start_frame
                    idx_j = self.WINDOW_SIZE - feat.start_frame
                    if 0 <= idx_i < len(feat.feature_per_frame) and \
                       0 <= idx_j < len(feat.feature_per_frame):
                        pts0.append(feat.feature_per_frame[idx_i][:2])
                        pts1.append(feat.feature_per_frame[idx_j][:2])
            if len(pts0) < 20:
                continue
            pts0 = np.array(pts0, dtype=np.float64)
            pts1 = np.array(pts1, dtype=np.float64)
            ok, R, t = solve_relative_pose(pts0, pts1)
            if ok and np.linalg.norm(t) > 0.1:
                return i, R, t
        return -1, np.eye(3), np.zeros(3)

    # ------------------------------------------------------------------ #
    #  Nonlinear optimisation                                              #
    # ------------------------------------------------------------------ #
    def _solve_odometry(self):
        if self.frame_count < self.WINDOW_SIZE and self.solver_flag == SolverFlag.INITIAL:
            return
        self.feature_manager.triangulate(self.Ps, self.Rs, self.tic[0], self.ric[0])
        self._backend_optimisation()

    def _backend_optimisation(self):
        """Gauss-Newton style optimisation via scipy least_squares."""
        # Build compact state vector
        N = self.frame_count + 1
        # State: for each frame [P(3), angle_axis(3), V(3), Ba(3), Bg(3)] = 15
        # Feature inverse depths
        feat_list = [(fid, feat) for fid, feat in self.feature_manager.feature.items()
                     if feat.estimated_depth > 0 and
                        feat.start_frame + len(feat.feature_per_frame) - 1 <= self.frame_count]

        def pack_state():
            x = []
            from scipy.spatial.transform import Rotation
            for i in range(N):
                x.extend(self.Ps[i].tolist())
                aa = Rotation.from_matrix(self.Rs[i]).as_rotvec()
                x.extend(aa.tolist())
                x.extend(self.Vs[i].tolist())
                x.extend(self.Bas[i].tolist())
                x.extend(self.Bgs[i].tolist())
            for _, feat in feat_list:
                x.append(1.0 / feat.estimated_depth)
            return np.array(x)

        def unpack_state(x):
            from scipy.spatial.transform import Rotation
            Ps_l, Rs_l, Vs_l, Bas_l, Bgs_l = [], [], [], [], []
            for i in range(N):
                base = i * 15
                Ps_l.append(x[base:base+3])
                Rs_l.append(Rotation.from_rotvec(x[base+3:base+6]).as_matrix())
                Vs_l.append(x[base+6:base+9])
                Bas_l.append(x[base+9:base+12])
                Bgs_l.append(x[base+12:base+15])
            inv_depths = x[N*15:]
            return Ps_l, Rs_l, Vs_l, Bas_l, Bgs_l, inv_depths

        def residuals(x):
            Ps_l, Rs_l, Vs_l, Bas_l, Bgs_l, inv_depths = unpack_state(x)
            res = []

            # IMU residuals
            for i in range(1, N):
                pre = self.pre_integrations[i]
                if pre is None:
                    continue
                Qi = normalize_quat(rot_to_quat(Rs_l[i-1]))
                Qj = normalize_quat(rot_to_quat(Rs_l[i]))
                r = pre.evaluate(Ps_l[i-1], Qi, Vs_l[i-1], Bas_l[i-1], Bgs_l[i-1],
                                  Ps_l[i], Qj, Vs_l[i], Bas_l[i], Bgs_l[i], self.gravity)
                res.extend(r.tolist())

            # Visual residuals
            for k, (fid, feat) in enumerate(feat_list):
                if k >= len(inv_depths):
                    break
                inv_d = inv_depths[k]
                if inv_d <= 0:
                    continue
                idx0 = feat.start_frame
                obs0 = feat.feature_per_frame[0]
                Qi = normalize_quat(rot_to_quat(Rs_l[idx0]))
                qic0 = normalize_quat(rot_to_quat(self.ric[0]))
                for j, obs in enumerate(feat.feature_per_frame[1:]):
                    idx_j = idx0 + j + 1
                    if idx_j >= N:
                        break
                    Qj = normalize_quat(rot_to_quat(Rs_l[idx_j]))
                    r = project_mono(obs0[:3], obs[:3],
                                     Ps_l[idx0], Qi, Ps_l[idx_j], Qj,
                                     self.tic[0], qic0, inv_d)
                    res.extend(r.tolist())

            if not res:
                return np.zeros(1)
            return np.array(res)

        x0 = pack_state()
        try:
            result = least_squares(
                residuals, x0, method='lm',
                max_nfev=self.params.max_num_iterations * 10,
                ftol=1e-4, xtol=1e-4,
            )
        except Exception:
            return

        Ps_l, Rs_l, Vs_l, Bas_l, Bgs_l, inv_depths = unpack_state(result.x)
        for i in range(N):
            self.Ps[i] = Ps_l[i]
            self.Rs[i] = Rs_l[i]
            self.Vs[i] = Vs_l[i]
            self.Bas[i] = Bas_l[i]
            self.Bgs[i] = Bgs_l[i]
        for k, (fid, feat) in enumerate(feat_list):
            if k < len(inv_depths) and inv_depths[k] > 0:
                feat.estimated_depth = 1.0 / inv_depths[k]

    # ------------------------------------------------------------------ #
    #  Sliding window management                                           #
    # ------------------------------------------------------------------ #
    def _slide_window(self):
        if self.margin_flag == MarginalizationFlag.MARGIN_OLD:
            self._slide_window_old()
        else:
            self._slide_window_new()

    def _slide_window_old(self):
        if self.solver_flag == SolverFlag.NON_LINEAR:
            # Shift all states back by one
            for i in range(self.WINDOW_SIZE):
                self.Rs[i] = self.Rs[i+1].copy()
                self.Ps[i] = self.Ps[i+1].copy()
                self.Vs[i] = self.Vs[i+1].copy()
                self.Bas[i] = self.Bas[i+1].copy()
                self.Bgs[i] = self.Bgs[i+1].copy()
                self.pre_integrations[i] = self.pre_integrations[i+1]
            self.pre_integrations[self.WINDOW_SIZE] = IntegrationBase(
                self.acc_0, self.gyr_0,
                self.Bas[self.WINDOW_SIZE], self.Bgs[self.WINDOW_SIZE],
                self.params.acc_n, self.params.gyr_n,
                self.params.acc_w, self.params.gyr_w,
            )
        else:
            if self.frame_count == self.WINDOW_SIZE:
                for i in range(self.WINDOW_SIZE):
                    self.Rs[i] = self.Rs[i+1].copy()
                    self.Ps[i] = self.Ps[i+1].copy()
                    self.Vs[i] = self.Vs[i+1].copy()
                    self.Bas[i] = self.Bas[i+1].copy()
                    self.Bgs[i] = self.Bgs[i+1].copy()
                    self.pre_integrations[i] = self.pre_integrations[i+1]

    def _slide_window_new(self):
        if self.solver_flag != SolverFlag.NON_LINEAR:
            return
        if self.frame_count == self.WINDOW_SIZE:
            self.pre_integrations[self.WINDOW_SIZE - 1] = self.pre_integrations[self.WINDOW_SIZE]
            self.Rs[self.WINDOW_SIZE - 1] = self.Rs[self.WINDOW_SIZE].copy()
            self.Ps[self.WINDOW_SIZE - 1] = self.Ps[self.WINDOW_SIZE].copy()
            self.Vs[self.WINDOW_SIZE - 1] = self.Vs[self.WINDOW_SIZE].copy()
            self.Bas[self.WINDOW_SIZE - 1] = self.Bas[self.WINDOW_SIZE].copy()
            self.Bgs[self.WINDOW_SIZE - 1] = self.Bgs[self.WINDOW_SIZE].copy()

    # ------------------------------------------------------------------ #
    #  Failure detection                                                   #
    # ------------------------------------------------------------------ #
    def _failure_detection(self) -> bool:
        if len(self.feature_manager.feature) < 5:
            return True
        P = self.Ps[self.WINDOW_SIZE]
        if np.linalg.norm(P) > 200.0:
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Output                                                              #
    # ------------------------------------------------------------------ #
    def _get_odometry(self, header: float) -> Optional[Dict]:
        if self.solver_flag != SolverFlag.NON_LINEAR:
            return None
        P = self.Ps[self.frame_count].copy()
        Q = normalize_quat(rot_to_quat(self.Rs[self.frame_count]))
        V = self.Vs[self.frame_count].copy()
        Ba = self.Bas[self.frame_count].copy()
        Bg = self.Bgs[self.frame_count].copy()
        return {
            "timestamp": header,
            "position": P,
            "orientation": Q,  # [w,x,y,z]
            "velocity": V,
            "acc_bias": Ba,
            "gyr_bias": Bg,
        }
