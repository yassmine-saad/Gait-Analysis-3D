import cv2
import numpy as np
from robot.follow_controller import FollowController
from camera.camera import Camera
from vision.pose_estimator import PoseEstimator
from vision.GaitAnalyzer import GaitAnalyzer
from visualisation.visualizer import Visualizer
from exporter.exporter import Exporter

from utils.utils import (
    get_pose_model_alias,
    detect_person,  
)


def main():

    import time

    last_robot_cmd_time = 0.0
    ROBOT_CMD_PERIOD = 0.1  # seconds → 10 Hz
    # -------------------------------------------------------
    # 1) CAMERA INITIALIZATION
    # -------------------------------------------------------
    print("[MAIN] Starting RealSense...")
    cam = Camera(calib_file_path="camera/camera_calibration.npz")
    cam.start()

    # -------------------------------------------------------
    # 2) POSE ESTIMATOR INITIALIZATION
    # -------------------------------------------------------
    print("[MAIN] Initializing PoseEstimator (HRNet WholeBody)...")
    model_alias = get_pose_model_alias()
    pose_estimator = PoseEstimator(model_alias=model_alias, device="cuda")
    follower = FollowController()

    # -------------------------------------------------------
    # 3) LIVE LOOP VARIABLES
    # -------------------------------------------------------
    recording = False
    recorded_frames = []
    recorded_k3d_cam = []
    POSE_EVERY = 3  # try 3 first (2 if GPU is strong)
    frame_id = 0
    last_k2d = None

    print(
        "\n[MAIN] LIVE MODE STARTED\n"
        "  - Skeleton shown live when person is detected\n"
        "  - Press R to START recording\n"
        "  - Press S to STOP recording and analyze\n"
        "  - Press Q or ESC to quit\n"
    )

    # -------------------------------------------------------
    # 4) LIVE LOOP
    # -------------------------------------------------------
    prev_time = cv2.getTickCount()
    last_valid_distance = None

    while True:

        # -------------------------------
        # Get camera frames
        # -------------------------------
        rgb, depth = cam.get_frames()
        if rgb is None or depth is None:
            continue

        draw_img = rgb.copy()
        curr_time = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(
            draw_img,
            f"FPS: {fps:.1f}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        # -------------------------------
        # PERSON DETECTION (YOLO ONLY)
        # -------------------------------
        bboxes = detect_person(rgb)
        person_detected = len(bboxes) > 0

        # -------------------------------
        # POSE ESTIMATION (FULL IMAGE)
        # -------------------------------
        k2d = None
        frame_id += 1

        if person_detected and (frame_id % POSE_EVERY == 0 or last_k2d is None):
            H, W = rgb.shape[:2]
            full_bbox = (0, 0, W, H)
            rgb_for_pose = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

            k2d = pose_estimator.estimate_2d(rgb_for_pose, full_bbox)
            if k2d is not None:
                last_k2d = k2d
        else:
            k2d = last_k2d

        # -------------------------------
        # SUBJECT POSITION FOR FOLLOWING
        # -------------------------------
        subject_distance = None
        subject_offset_x = None

        if k2d is not None:
            # Use pelvis center (stable)
            L_HIP, R_HIP = 1, 7

            u = int((k2d[L_HIP, 0] + k2d[R_HIP, 0]) / 2)
            v = int((k2d[L_HIP, 1] + k2d[R_HIP, 1]) / 2)

            image_center_x = rgb.shape[1] // 2
            subject_offset_x = u - image_center_x

            if 0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]:
                window = depth[max(0, v - 3) : v + 3, max(0, u - 3) : u + 3]
                valid = window[(window > 0.4) & (window < 5.0)]

                if valid.size > 0:
                    subject_distance = float(np.median(valid))
                else:
                    subject_distance = None
        print(f"[DEBUG] subject_distance = {subject_distance}")

        # -------------------------------
        # ROBOT FOLLOW COMMAND
        # -------------------------------

        now = time.time()

        if now - last_robot_cmd_time >= ROBOT_CMD_PERIOD:
            if subject_distance is not None:
                follower.update(subject_distance, subject_offset_x)
            else:
                follower.stop()

            last_robot_cmd_time = now
        if subject_distance is not None:
            last_valid_distance = subject_distance
        else:
            subject_distance = last_valid_distance

        # -------------------------------
        # DRAW LIVE SKELETON
        # -------------------------------
        if k2d is not None:
            for u, v in k2d.astype(int):
                cv2.circle(draw_img, (u, v), 4, (0, 255, 255), -1)

        # -------------------------------
        # UI TEXT
        # -------------------------------
        if recording:
            cv2.putText(
                draw_img,
                "RECORDING... (Press S to stop)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
        else:
            if person_detected:
                cv2.putText(
                    draw_img,
                    "PERSON DETECTED (Press R to record)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
            else:
                cv2.putText(
                    draw_img,
                    "NO PERSON DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 255),
                    2,
                )

        # -------------------------------
        # SHOW LIVE VIEW
        # -------------------------------
        cv2.imshow("Live Gait View", draw_img)
        key = cv2.waitKey(1) & 0xFF

        # -------------------------------
        # QUIT
        # -------------------------------
        if key in [27, ord("q")]:
            print("\n[MAIN] Quit command received. Exiting.")
            follower.stop()
            cam.stop()
            cv2.destroyAllWindows()
            return

        # -------------------------------
        # START RECORDING
        # -------------------------------
        if key == ord("r"):
            if not person_detected:
                print("[MAIN] Cannot start recording — no person detected.")
            else:
                print("[MAIN] Recording started.")
                follower.stop()
                recording = True
                recorded_frames = []

        # -------------------------------
        # STOP RECORDING
        # -------------------------------
        if key == ord("s"):
            if recording:
                print("[MAIN] Recording stopped. Proceeding to analysis.")
                follower.stop()
                recording = False
                break
            else:
                print("[MAIN] Not currently recording.")

        # -------------------------------
        # SAVE FRAMES (ONLY IF RECORDING)
        # -------------------------------
        if recording and k2d is not None:

            intrinsics, _, _ = cam.get_intrinsics()
            k3d = pose_estimator.convert_to_3d(k2d, depth, intrinsics)
            if k3d is None:
                continue

            k3d_sag = pose_estimator.convert_to_sagittal(k3d)

            recorded_frames.append(k3d_sag)  # was k3d_smooth
            recorded_k3d_cam.append(k3d.copy())
    # -------------------------------------------------------
    # 5) ANALYSIS PHASE
    # -------------------------------------------------------
    cam.stop()
    cv2.destroyAllWindows()

    if len(recorded_frames) < 30:
        print(
            f"[MAIN] ERROR: Sequence too short ({len(recorded_frames)} frames). "
            "Need at least 30 frames."
        )
        return

    sequence = np.array(recorded_frames, dtype=np.float32)
    print(f"[MAIN] Recorded frames: {sequence.shape[0]}")
    # OFFLINE smoothing (Kalman + RTS) on keypoints BEFORE angles
    sequence = pose_estimator.smooth_sequence_offline(sequence, q=1e-4, r=5e-3)
    print("[MAIN] Offline smoothing applied.")
    # -------------------------------------------------------
    # 6) GAIT ANALYSIS
    # -------------------------------------------------------
    analyzer = GaitAnalyzer()
    results = analyzer.analyze_sequence(sequence)

    # -------------------------------------------------------
    # 7) VISUALIZATION
    # -------------------------------------------------------
    visualizer = Visualizer()
    visualizer.plot_all(results)

    # -------------------------------------------------------
    # 8) EXPORT RESULTS
    # -------------------------------------------------------
    exporter = Exporter(visualizer.person_dir)
    exporter.export_keypoints(sequence, filename="keypoints_3d.npy")
    exporter.export_curves_csv(results, filename="gait_curves.csv")
    exporter.export_pdf_report(results, filename="report.pdf")
    exporter.export_camera_joints_csv(
        recorded_k3d_cam, fps=30, filename="joints_camera_frame.csv"
    )

    print("\n=============================================")
    print("   GAIT ANALYSIS COMPLETE")
    print("   Results saved in:")
    print("   ", visualizer.person_dir)
    print("=============================================")
    print("Recorded frames:", len(recorded_frames))


if __name__ == "__main__":
    main()
