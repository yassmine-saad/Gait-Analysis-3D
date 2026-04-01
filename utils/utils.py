import os
import cv2
import numpy as np
from datetime import datetime
from ultralytics import YOLO

# ===============================================================
# YOLOv8 PERSON DETECTOR (PYTORCH CUDA)
# ===============================================================

_yolo_model = YOLO("yolov8n.pt")
_yolo_model.to("cuda")   # FORCE GPU


def detect_person(image):
    """
    Detect persons using YOLOv8 (PyTorch CUDA).
    Returns list of bounding boxes: (x, y, w, h)
    """

    results = _yolo_model(
        image,
        device=0,        # CUDA
        conf=0.35,
        classes=[0],     # person only
        verbose=False
    )

    if not results or results[0].boxes is None:
        return []

    final_boxes = []
    for b in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = b
        final_boxes.append((
            int(x1),
            int(y1),
            int(x2 - x1),
            int(y2 - y1)
        ))

    return final_boxes

# ===============================================================
# POSE MODEL ALIAS
# ===============================================================

def get_pose_model_alias():
    return "td-hm_hrnet-w48_8xb32-210e_coco-wholebody-384x288"


# ===============================================================
# GAIT SEQUENCE CAPTURE
# ===============================================================

def capture_gait_sequence(
    pose_estimator,
    camera,
    max_frames=300,
    min_frames_required=30
):
    """
    Capture sagittal 3D gait sequence.
    """

    sequence = []

    print("\n[Utils] Gait capture started...")

    while len(sequence) < max_frames:

        rgb, depth = camera.get_frames()
        if rgb is None or depth is None:
            break

        # ----------------------------
        # YOLO PERSON CHECK
        # ----------------------------
        if len(detect_person(rgb)) == 0:
            continue

        # ----------------------------
        # Pose estimation
        # ----------------------------
        k2d = pose_estimator.estimate_2d(rgb)
        if k2d is None:
            continue

        intrinsics, _, _ = camera.get_intrinsics()
        k3d = pose_estimator.convert_to_3d(k2d, depth, intrinsics)
        if k3d is None:
            continue

        k3d_sag = pose_estimator.convert_to_sagittal(k3d)
        mid_hip = 0.5 * (k3d_sag[1] + k3d_sag[7])
        if np.isnan(mid_hip).any():
            continue

        sequence.append(k3d_sag)

    if len(sequence) < min_frames_required:
        return None

    print(f"[Utils] Captured {len(sequence)} frames.")
    return np.array(sequence, dtype=np.float32)


# ===============================================================
# OUTPUT FOLDER
# ===============================================================

def create_output_folder(base="results"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = os.path.join(base, f"session_{ts}")
    os.makedirs(folder, exist_ok=True)
    return folder
