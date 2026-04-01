import pyrealsense2 as rs
import numpy as np
import cv2

class Camera:
    def __init__(self, calib_file_path=None, apply_smooth=False, smooth_window=5):
        """
        Wrapper class for the Intel RealSense D435i camera.

        Args:
            calib_file_path (str): Path to .npz with 'camera_matrix' + 'dist_coeffs'.
            apply_smooth (bool): Apply temporal smoothing to RGB + depth.
            smooth_window (int): Number of frames to average for smoothing.
        """
        # RealSense pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable streams
        self.config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 60)
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 60)

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Frame buffers
        self.frame_rgb = None
        self.frame_depth = None

        # Calibration values
        self.intrinsics = None
        self.dist_coeffs = None
        self.depth_scale = None

        # Smoothing
        self.apply_smooth = apply_smooth
        self.smooth_window = smooth_window
        self.frame_rgb_history = []
        self.frame_depth_history = []

        # Load calibration if provided
        if calib_file_path:
            self._load_calibration(calib_file_path)

    def _load_calibration(self, path):
        """Load saved camera intrinsics from .npz file."""
        try:
            calib = np.load(path)
            self.intrinsics = calib["camera_matrix"]
            self.dist_coeffs = calib["dist_coeffs"]
            print(f"[Camera] Calibration loaded from {path}.")
        except Exception as e:
            print(f"[Camera] ERROR loading calibration file: {e}")

    def start(self):
        """Start the RealSense camera pipeline."""
        profile = self.pipeline.start(self.config)

        # Fetch depth scale
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        print(f"[Camera] Camera started. Depth scale: {self.depth_scale} meters/unit.")

        # If no calibration file was provided, use RealSense intrinsics
        if self.intrinsics is None:
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

            self.intrinsics = np.array([
                [intr.fx,     0,      intr.ppx],
                [0,        intr.fy,   intr.ppy],
                [0,           0,         1    ]
            ], dtype=np.float32)

            self.dist_coeffs = np.array(intr.coeffs, dtype=np.float32)
            print("[Camera] Loaded intrinsics directly from RealSense device.")

    def stop(self):
        """Stop camera streaming."""
        self.pipeline.stop()
        print("[Camera] Camera stopped.")

    def get_frames(self):
        """
        Retrieve the latest aligned RGB and depth frame.

        Returns:
            frame_rgb (np.ndarray): BGR color frame (sharp)
            frame_depth (np.ndarray): depth frame in meters (smoothed if enabled)
        """
        frames = self.pipeline.wait_for_frames()

        # Align depth to color
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        # Convert to NumPy arrays
        self.frame_rgb = np.asanyarray(color_frame.get_data())
        self.frame_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32) * self.depth_scale

        # Optional smoothing on depth only
        if self.apply_smooth:
            self.frame_depth = cv2.GaussianBlur(self.frame_depth, (5, 5), 0)

        return self.frame_rgb, self.frame_depth

    def get_intrinsics(self):
        """
        Returns:
            intrinsics (np.ndarray): 3x3 intrinsic camera matrix
            dist_coeffs (np.ndarray): distortion coefficients
            depth_scale (float): scale factor for depth values
        """
        return self.intrinsics, self.dist_coeffs, self.depth_scale

    def apply_smoothing(self, enable=True, target='depth'):
        """
        Enable or disable frame smoothing.

        Args:
            enable (bool): If True, smoothing is applied in get_frames().
            target (str): 'depth' or 'rgb' or 'both'
        """
        self.apply_smooth = enable
        self.smooth_target = target
        state = "enabled" if enable else "disabled"
        print(f"[Camera] Smoothing {state} for: {target}.")
