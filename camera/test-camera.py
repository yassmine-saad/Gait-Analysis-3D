import cv2
from camera import Camera

# Calibration file
calib_file = r"C:\Users\LOQ\OneDrive\Bureau\Projet semestriel\camera\camera_calibration.npz"

# Initialize camera
cam = Camera(calib_file_path=calib_file)

# Enable depth smoothing only (RGB stays sharp)
cam.apply_smoothing(enable=False, target='both')

# Start camera
cam.start()

try:
    while True:
        frame_rgb, frame_depth = cam.get_frames()

        # Show RGB frame (sharp)
        cv2.imshow("RGB Frame", frame_rgb)

        # Visualize depth (smoothed)
        if frame_depth is not None and frame_depth.max() > 0:
            depth_vis = cv2.convertScaleAbs(frame_depth, alpha=255.0 / frame_depth.max())
            cv2.imshow("Depth Frame", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cam.stop()
    cv2.destroyAllWindows()

# Print intrinsics
camera_matrix, dist_coeffs, depth_scale = cam.get_intrinsics()
print("\n===== Camera Intrinsics =====")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coeffs:\n", dist_coeffs)
print("Depth Scale:", depth_scale)
