import numpy as np
from scipy.signal import savgol_filter, find_peaks


class GaitAnalyzer:
    """
    Offline gait analysis based on sagittal-plane kinematics.

    Input:
        keypoints_3d_seq: shape (N_frames, 12, 3)
        Sagittal frame:
            x = lateral
            y = forward
            z = up
    """

    def __init__(self, use_smoothing=True, smooth_window=31, smooth_poly=3, fps=30):

        self.use_smoothing = use_smoothing
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.fps = fps

        self.gait_phase = None
        self.angles_left = None
        self.angles_right = None
        self.metrics = {}
        self.flags = []

    # ======================================================
    # MAIN ENTRY
    # ======================================================
    def analyze_sequence(self, keypoints_3d_seq):
        kp = np.asarray(keypoints_3d_seq, dtype=np.float32)
        N, J, _ = kp.shape

        if J != 12:
            raise ValueError("Expected 12 joints from PoseEstimator")

        # LEFT
        L_hip = kp[:, 1]
        L_knee = kp[:, 2]
        L_ank = kp[:, 3]
        L_heel = kp[:, 4]
        L_toe = kp[:, 5]
        # RIGHT
        R_hip = kp[:, 7]
        R_knee = kp[:, 8]
        R_ank = kp[:, 9]
        R_heel = kp[:, 10]
        R_toe = kp[:, 11]

        angles_L = self._compute_leg_angles(L_hip, L_knee, L_ank, L_toe, L_heel)
        angles_R = self._compute_leg_angles(R_hip, R_knee, R_ank, R_toe, R_heel)

        strides = self._segment_gait_cycles(L_ank, R_ank)

        if len(strides) == 0:
            self.flags.append("No valid gait cycle detected")
            self._fallback_curves(angles_L, angles_R)
            self.metrics = self._compute_internal_metrics()
            return self._package()

        phase_target = np.linspace(0, 100, 101)

        def collect(angles):
            curves = {"hip": [], "knee": [], "ankle": []}
            for s, e in strides:
                if e - s < 5:
                    continue
                local_phase = np.linspace(0, 100, e - s)
                for j in curves:
                    curves[j].append(
                        np.interp(phase_target, local_phase, angles[j][s:e])
                    )
            return curves

        Lc = collect(angles_L)
        Rc = collect(angles_R)
        if (len(Lc["hip"]) == 0) or (len(Rc["hip"]) == 0):
            self.flags.append(
                "Gait cycles detected but none were usable after filtering"
            )
            self._fallback_curves(angles_L, angles_R)
            self.metrics = self._compute_internal_metrics()
            return self._package()

        self.gait_phase = phase_target
        self.angles_left = {k: np.mean(v, axis=0) for k, v in Lc.items()}
        self.angles_right = {k: np.mean(v, axis=0) for k, v in Rc.items()}
        # Zero angles at mid-stance (50% gait)
        mid_idx = np.argmin(np.abs(self.gait_phase - 50))
        for joint in ["hip", "knee", "ankle"]:
            ref = self.angles_left[joint][mid_idx]
            self.angles_left[joint] -= ref
            self.angles_right[joint] -= ref
        """
        for joint in self.angles_left:
            zero = self.angles_left[joint][mid_idx]
            self.angles_left[joint] -= zero
            self.angles_right[joint] -= zero
        """
        """"
        for side in (self.angles_left, self.angles_right):
            for joint in side:
                side[joint] = side[joint] - side[joint][mid_idx]
        """
        ###################
        if self.use_smoothing:
            for side in (self.angles_left, self.angles_right):
                for k in side:
                    side[k] = self._smooth_curve(side[k])
        self.metrics = self._compute_internal_metrics()
        return self._package()

    # ======================================================
    # ✅ BIOMECHANICALLY CORRECT ANGLES
    # ======================================================
    def _compute_leg_angles(self, hip, knee, ankle, toe, heel):
        """
        Biomechanically correct sagittal-plane joint angles (degrees).

        Frame convention:
            x = lateral
            y = forward
            z = up

        Sign convention:
            Hip:  flexion (+), extension (-)
            Knee: flexion (+), 0° = full extension
            Ankle: dorsiflexion (+), plantarflexion (-)
        """

        # --- Project points to sagittal plane (y, z) ---
        H = hip[:, 1:3]  # (forward, up)
        K = knee[:, 1:3]
        A = ankle[:, 1:3]

        # --- Segment vectors ---
        thigh = K - H  # hip -> knee
        shank = A - K  # knee -> ankle

        # --- Reference vertical axis ---
        vertical = np.array([0.0, 1.0], dtype=np.float32)

        # --- Helper: signed angle between 2D vectors ---
        def signed_angle(u, v):
            """
            Signed angle between 2D vectors.
            u : (N,2)
            v : (N,2) or (2,)
            """

            u = np.asarray(u, dtype=np.float32)
            v = np.asarray(v, dtype=np.float32)

            # If v is constant, broadcast it
            if v.ndim == 1:
                v = np.tile(v, (u.shape[0], 1))

            cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
            dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]

            return np.degrees(np.arctan2(cross, dot))

        # ======================
        # HIP FLEXION / EXTENSION
        # ======================
        # angle between thigh and vertical
        hip_angle = signed_angle(thigh, vertical)

        # ======================
        # KNEE FLEXION / EXTENSION
        # ======================

        # angle between thigh and shank
        # full extension -> 0°, flexion positive
        knee_angle = 180.0 - self._angle_between_series(thigh, shank)
        # knee_angle = knee_angle - knee_angle[0]

        # ======================
        # ANKLE DORSI / PLANTAR
        # ======================

        # Foot segment: ankle -> midpoint(heel, toe)
        foot_center = (toe[:, 1:3] + heel[:, 1:3]) / 2.0
        foot = foot_center - ankle[:, 1:3]
        # Normalize foot to reduce jitter
        foot_norm = np.linalg.norm(foot, axis=1, keepdims=True)
        foot = foot / np.clip(foot_norm, 1e-6, None)
        # Shank segment: knee -> ankle
        shank_2d = ankle[:, 1:3] - knee[:, 1:3]

        ankle_angle = signed_angle(foot, shank_2d)
        ankle_angle = self._unwrap_only(ankle_angle)

        # --- Unwrap + zero to initial posture ---
        hip_angle = self._unwrap_only(hip_angle)
        """
        # --------------------------------------------------
        # Fix mirrored sagittal sign for right leg
        # --------------------------------------------------
        if np.mean(ankle_angle) < -90:
            ankle_angle = ankle_angle + 180.0

        if np.mean(hip_angle) < -90:
            hip_angle = hip_angle + 180.0

        """
        return {
            "hip": hip_angle,
            "knee": knee_angle,
            "ankle": ankle_angle,
        }

    # ======================================================
    # ANGLE HELPERS
    # ======================================================
    def _angle_between_series(self, u, v):
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)
        dot = np.clip(np.sum(u * v, axis=1), -1.0, 1.0)
        return np.degrees(np.arccos(dot))

    def _unwrap_only(self, a):
        a = np.unwrap(np.radians(a))
        return np.degrees(a)

    def _unwrap_and_zero(self, a):
        a = np.unwrap(np.radians(a))
        a = np.degrees(a)
        return a - a[0]

    # ======================================================
    # UTILS
    # ======================================================

    def _smooth_curve(self, y):
        n = len(y)
        if n < 5:
            return y
        win = min(self.smooth_window, n // 2 * 2 - 1)
        win = max(win, 3)
        poly = min(self.smooth_poly, win - 1)
        return savgol_filter(y, win, poly)

    def _segment_gait_cycles(self, L_ank, R_ank):
        # Z = 0.5 * (L_ank[:, 2] + R_ank[:, 2])
        Z = L_ank[:, 2]  # LEFT ankle vertical motion ONLY

        Zs = self._smooth_curve(Z)
        peaks, _ = find_peaks(-Zs, distance=int(self.fps * 0.4))
        # print(f"[GAIT] Detected strides: {len(peaks)-1}")

        return [(peaks[i], peaks[i + 1]) for i in range(len(peaks) - 1)]

    def _fallback_curves(self, L, R):
        phase = np.linspace(0, 100, 101)
        raw = np.linspace(0, 100, len(L["hip"]))
        self.gait_phase = phase
        self.angles_left = {k: np.interp(phase, raw, L[k]) for k in L}
        self.angles_right = {k: np.interp(phase, raw, R[k]) for k in R}
        if self.use_smoothing:
            for side in (self.angles_left, self.angles_right):
                for k in side:
                    side[k] = self._smooth_curve(side[k])

    # ======================================================
    # METRICS
    # =====================================================
    def _compute_internal_metrics(self):
        metrics = {}
        for side, ang in [("left", self.angles_left), ("right", self.angles_right)]:
            metrics[side] = {
                "rom_hip": float(np.ptp(ang["hip"])),
                "rom_knee": float(np.ptp(ang["knee"])),
                "rom_ankle": float(np.ptp(ang["ankle"])),
            }
        # ---------------------------------
        # Left–Right ROM difference
        # (Potential anomaly indicator)
        # ---------------------------------
        metrics["rom_diff"] = {
            "hip": abs(metrics["left"]["rom_hip"] - metrics["right"]["rom_hip"]),
            "knee": abs(metrics["left"]["rom_knee"] - metrics["right"]["rom_knee"]),
            "ankle": abs(metrics["left"]["rom_ankle"] - metrics["right"]["rom_ankle"]),
        }
        return metrics

    def _package(self):
        return {
            "gait_phase": self.gait_phase,
            "angles_left": self.angles_left,
            "angles_right": self.angles_right,
            "metrics": self.metrics,
            "flags": self.flags,
        }
