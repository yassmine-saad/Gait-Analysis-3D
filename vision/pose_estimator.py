import numpy as np
from filterpy.kalman import KalmanFilter
from mmpose.apis import MMPoseInferencer


class PoseEstimator:
    """
    Real-time whole-body pose estimator using a COCO-WholeBody model (133 keypoints).

    Extracted biomechanical joints (12 total) – internal order:

        LEFT side
        ---------------------------
        0: LShoulder   [body 5]
        1: LHip        [body 11]
        2: LKnee       [body 13]
        3: LAnkle      [body 15]
        4: LHeel       [foot 19]
        5: LBigToe     [foot 17]

        RIGHT side
        ---------------------------
        6: RShoulder   [body 6]
        7: RHip        [body 12]
        8: RKnee       [body 14]
        9: RAnkle      [body 16]
        10: RHeel      [foot 22]
        11: RBigToe    [foot 20]

    Pipeline:
        - 2D detection (MMPose whole-body)
        - Depth projection to 3D (RealSense)
        - Smoothed person-aligned sagittal frame
        - Homogeneous transform camera → sagittal
        - Kalman smoothing on 12 joints
    """

    # Indices in COCO-WholeBody array
    KEYPOINT_INDICES = [
        5,
        11,
        13,
        15,
        19,
        17,  # Left: shoulder, hip, knee, ankle, heel, toe
        6,
        12,
        14,
        16,
        22,
        20,  # Right: shoulder, hip, knee, ankle, heel, toe
    ]

    def __init__(
        self,
        model_alias="td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288",
        device="cuda",
        camera_orientation="front",  # unused but kept for compatibility
    ):
        self.device = device
        self.camera_orientation = camera_orientation

        self.inferencer = MMPoseInferencer(pose2d=model_alias, device=device)

        # Joint buffers
        self.keypoints_2d = None
        self.keypoints_3d_cam = None
        self.keypoints_3d_sag = None

        # Sagittal transform
        self.T_cam_to_sag = None
        self.smooth_R = None  # smoothed rotation matrix for axes

        # EMA buffers for anchor joints (shoulders + hips)
        self.smooth_L_sh = None
        self.smooth_R_sh = None
        self.smooth_L_hip = None
        self.smooth_R_hip = None

        # Kalman filters for 12 joints × 3 coords
        self.kalman_filters = [self._init_kalman() for _ in range(36)]
        self.initialized_sag = False
        self.frame_count = 0

    # -------------------------
    # 2D ESTIMATION
    # -------------------------
    def estimate_2d(self, rgb_frame, bbox=None):
        """
        Run top-down pose estimation using YOLO bbox.
        bbox format: (x, y, w, h) in full image coordinates
        """
        if bbox is None:
            H, W = rgb_frame.shape[:2]
            bbox = (0, 0, W, H)

        x, y, w, h = bbox
        det_results = [
            {
                "bbox": np.array([x, y, x + w, y + h], dtype=np.float32),
                "bbox_score": 1.0,
                "category_id": 0,  # person
            }
        ]

        try:
            result = next(self.inferencer(rgb_frame, det_results=det_results))
        except Exception as e:
            print("[PoseEstimator] Pose inference failed:", e)
            return None

        preds = result.get("predictions", None)
        if preds is None or len(preds) == 0:
            return None

        # 🔥 THIS IS THE CRITICAL FIX
        person = preds[0]  # already the person dict
        if isinstance(person, list):
            if len(person) == 0:
                return None
            person = person[0]
        if not isinstance(person, dict) or "keypoints" not in person:
            return None
        kpts = np.asarray(person["keypoints"], dtype=np.float32)

        if kpts.ndim != 2 or kpts.shape[0] < max(self.KEYPOINT_INDICES) + 1:
            return None

        selected = kpts[self.KEYPOINT_INDICES, :2]
        self.keypoints_2d = selected
        return selected

    # -------------------------
    # 3D PROJECTION
    # -------------------------

    def convert_to_3d(self, keypoints_2d, depth_frame, intrinsics):

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        pts = []
        for idx, (u, v) in enumerate(keypoints_2d):
            x, y = int(u), int(v)

            # -----------------------------
            # Robust depth (median patch)
            # -----------------------------
            h, w = depth_frame.shape

            x0 = max(0, x - 2)
            x1 = min(w, x + 3)
            y0 = max(0, y - 2)
            y1 = min(h, y + 3)

            patch = depth_frame[y0:y1, x0:x1]
            valid = patch[patch > 0]

            if valid.size > 0:
                z = np.median(valid)
            else:
                if self.keypoints_3d_cam is not None:
                    pts.append(self.keypoints_3d_cam[idx])
                else:
                    pts.append([0, 0, 0])
                continue

            X = (x - cx) * z / fx
            Y = (y - cy) * z / fy
            pts.append([X, Y, z])

        self.keypoints_3d_cam = np.array(pts, dtype=np.float32)
        return self.keypoints_3d_cam

    # -------------------------
    # SAGITTAL FRAME WITH SMOOTHING
    # -------------------------
    def convert_to_sagittal(self, k3d):
        """
        Convert camera-frame 3D joints to a FIXED sagittal frame.
        The sagittal frame is calibrated ONCE (first stable frame).
        """
        self.frame_count += 1

        L_hip = k3d[1]
        R_hip = k3d[7]

        # ----------------------------------------
        # 1) FIRST INITIALIZATION (STATIC CALIBRATION)
        # ----------------------------------------
        if not self.initialized_sag and self.frame_count > 30:

            print("[PoseEstimator] Calibrating sagittal anatomical frame...")

            # ORIGIN of sagittal frame = mid-hip
            mid_hip = 0.5 * (L_hip + R_hip)

            # ----------------------------------------
            # AXIS 1 = vertical anatomical axis (global up)
            # RealSense convention: Y axis grows DOWN => vertical_up = (0, -1, 0)
            # ----------------------------------------
            z_sag = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            # ----------------------------------------
            # AXIS 2 = left→right direction (lateral)
            # ----------------------------------------
            x_sag = R_hip - L_hip
            x_sag /= np.linalg.norm(x_sag) + 1e-8

            # ----------------------------------------
            # AXIS 3 = forward direction (orthogonal)
            # ----------------------------------------
            y_sag = np.cross(z_sag, x_sag)
            y_sag /= np.linalg.norm(y_sag) + 1e-8

            # Re-orthogonalize x to ensure perfect 90° axes
            x_sag = np.cross(y_sag, z_sag)
            x_sag /= np.linalg.norm(x_sag) + 1e-8

            # ----------------------------------------
            # Build rotation matrix R_sag
            # ----------------------------------------
            R = np.vstack([x_sag, y_sag, z_sag])

            # SVD orthonormalization
            U, _, Vt = np.linalg.svd(R)
            R = U @ Vt

            # ----------------------------------------
            # Compute translation such that mid-hip → origin
            # ----------------------------------------
            t = -R @ mid_hip

            # Build homogeneous transform
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = R
            T[:3, 3] = t

            self.T_cam_to_sag = T
            self.initialized_sag = True

        # ----------------------------------------
        # 2) APPLY FIXED TRANSFORM TO ALL JOINTS
        # ----------------------------------------
        if not self.initialized_sag:
         # Sagittal frame not ready yet → return raw camera-frame 3D
            self.keypoints_3d_sag = k3d
            return k3d
        pts_sag = []
        T = self.T_cam_to_sag

        for p in k3d:
            ph = np.array([p[0], p[1], p[2], 1.0], dtype=np.float32)
            pts_sag.append((T @ ph)[:3])

        pts_sag = np.array(pts_sag, dtype=np.float32)
        self.keypoints_3d_sag = pts_sag
        return pts_sag

    # -------------------------
    # KALMAN SMOOTHING (12×3)
    # -------------------------
    def apply_smoothing(self, pts):
        flat = pts.flatten()
        out = []

        for i, val in enumerate(flat):
            kf = self.kalman_filters[i]
            kf.predict()
            kf.update(val)
            out.append(kf.x[0])

        out = np.array(out, dtype=np.float32).reshape(pts.shape)
        self.keypoints_3d_sag = out
        return out

    # -------------------------
    def get_keypoints(self):
        return self.keypoints_2d, self.keypoints_3d_sag

    def _init_kalman(self):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        kf.x = np.array([0.0, 0.0])
        kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        kf.H = np.array([[1.0, 0.0]])
        kf.P *= 500.0
        kf.R = 0.01
        kf.Q = 0.001
        return kf

    def smooth_sequence_offline(self, seq_3d, q=1e-4, r=5e-3):
        """
        Offline smoothing for a recorded sequence using Kalman + RTS smoother.
        seq_3d: (N, J, 3) float32
        Returns: smoothed sequence same shape.
        """

        seq = np.asarray(seq_3d, dtype=np.float32)
        N, J, C = seq.shape
        out = np.empty_like(seq)

        dt = 1.0  # frame step (we work in frames; absolute dt not critical here)

        def make_kf(z0):
            kf = KalmanFilter(dim_x=2, dim_z=1)
            kf.F = np.array([[1, dt], [0, 1]], dtype=np.float32)
            kf.H = np.array([[1, 0]], dtype=np.float32)

            kf.x = np.array([[z0], [0]], dtype=np.float32)

            kf.P *= 1.0
            kf.R *= r
            kf.Q = np.array([[q, 0], [0, q]], dtype=np.float32)
            return kf

        for j in range(J):
            for c in range(C):
                z = seq[:, j, c].reshape(-1, 1)

                kf = make_kf(z0=z[0, 0])

                # batch filter + RTS smoother
                mu, cov, _, _ = kf.batch_filter(z)
                xs, _, _, _ = kf.rts_smoother(mu, cov)

                out[:, j, c] = xs[:, 0, 0]

        return out
