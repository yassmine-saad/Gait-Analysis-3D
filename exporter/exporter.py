import os
import numpy as np
import pandas as pd
from fpdf import FPDF


JOINT_NAMES = [
    "LShoulder",
    "LHip",
    "LKnee",
    "LAnkle",
    "LHeel",
    "LBigToe",
    "RShoulder",
    "RHip",
    "RKnee",
    "RAnkle",
    "RHeel",
    "RBigToe",
]

# ============================================================
# EXPORTER CLASS — Handles CSV, JSON, and PDF export
# ============================================================
class Exporter:
    """
    Exports gait-analysis results:
        - Curves CSV
        - JSON metrics
        - PDF report
        - Raw keypoint data
    """

    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    # --------------------------------------------------------
    # Export raw keypoints
    # --------------------------------------------------------
    def export_keypoints(self, keypoints_seq, filename="keypoints_3d.npy"):
        path = os.path.join(self.output_folder, filename)
        np.save(path, keypoints_seq)
        print(f"[Exporter] Saved keypoints → {path}")

    # --------------------------------------------------------
    # Export gait curves to CSV
    # --------------------------------------------------------
    def export_curves_csv(self, results, filename="gait_curves.csv"):
        phase = results["gait_phase"]
        L = results["angles_left"]
        R = results["angles_right"]

        df = pd.DataFrame(
            {
                "phase": phase,
                "L_hip": L["hip"],
                "L_knee": L["knee"],
                "L_ankle": L["ankle"],
                "R_hip": R["hip"],
                "R_knee": R["knee"],
                "R_ankle": R["ankle"],
            }
        )

        path = os.path.join(self.output_folder, filename)
        df.to_csv(path, index=False)
        print(f"[Exporter] Saved gait curves CSV → {path}")

    # --------------------------------------------------------
    # Export PDF Report
    # --------------------------------------------------------
    def export_pdf_report(self, results, filename="report.pdf"):
        metrics = results["metrics"]
        flags = results["flags"]

        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 10, "Gait Analysis Report", ln=True, align="C")
        pdf.ln(5)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, "Range of Motion (ROM) Summary", ln=True)
        pdf.ln(4)

        angles_L = results["angles_left"]
        angles_R = results["angles_right"]

        for side, angles in [("left", angles_L), ("right", angles_R)]:
            pdf.set_font("Arial", "B", 14)
            pdf.cell(0, 8, f"{side.capitalize()} Leg", ln=True)

            pdf.set_font("Arial", "", 12)

            for joint in ["hip", "knee", "ankle"]:
                a = np.asarray(angles[joint], dtype=np.float32)
                a_min = float(np.min(a))
                a_max = float(np.max(a))
                a_rom = a_max - a_min

                pdf.cell(
                    0,
                    6,
                    f"{joint.capitalize()} - "
                    f"min: {a_min:.2f}° | "
                    f"max: {a_max:.2f}° | "
                    f"ROM: {a_rom:.2f}°",
                    ln=True,
                )

            pdf.ln(2)

        pdf.ln(2)
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 8, "Flags (Potential Gait Anomalies)", ln=True)

        pdf.set_font("Arial", "", 12)
        if not flags:
            pdf.cell(0, 6, "No anomalies detected.", ln=True)
        else:
            for fl in flags:
                pdf.cell(0, 6, f"- {fl}", ln=True)

        path = os.path.join(self.output_folder, filename)
        pdf.output(path)
        print(f"[Exporter] Saved PDF report → {path}")

        # --------------------------------------------------------
    # Export camera-frame joint coordinates to CSV
    # --------------------------------------------------------
    def export_camera_joints_csv(
        self, k3d_cam_seq, fps, filename="joints_camera_frame.csv"
    ):
        """
        Export raw 3D joint coordinates in CAMERA reference frame.
        Shape: (N_frames, 12, 3)
        """

        import csv

        k3d_cam_seq = np.asarray(k3d_cam_seq, dtype=np.float32)

        header = ["frame", "time_sec"]
        for j in JOINT_NAMES:
            header += [f"{j}_Xcam", f"{j}_Ycam", f"{j}_Zcam"]

        path = os.path.join(self.output_folder, filename)

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for i, frame in enumerate(k3d_cam_seq):
                row = [i, i / fps]
                for joint in frame:
                    row.extend(joint.tolist())
                writer.writerow(row)

        print(f"[Exporter] Camera-frame joint CSV saved → {path}")
