import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class Visualizer:
    """
    Features:
    - Creates a timestamped folder per analyzed person
    - Plots ROM bar chart
    - All plots use normalized gait phase 0..100%
    """

    def __init__(self, base_dir="curves"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

        # Create unique folder using datetime
        self.person_dir = self._create_new_person_folder()

    # ------------------------------------------------------------------
    def _create_new_person_folder(self):
        """Creates a timestamped folder, e.g. curves/person_2025-12-10_18-42-11"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = os.path.join(self.base_dir, f"person_{timestamp}")
        os.makedirs(path, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    def plot_all(self, results: dict):
        """
        Plot all gait curves and save results.

        Args:
            results (dict): returned from GaitAnalyzer.analyze_sequence()
        """

        phase = results["gait_phase"]  # (101,) normalized 0..100
        angles_L = results["angles_left"]  # dict: hip/knee/ankle arrays (101,)
        angles_R = results["angles_right"]
        flags = results["flags"]
        metrics = results["metrics"]

        # --------------------------------------------------------------
        # Individual Joint Plots
        # --------------------------------------------------------------
        self._plot_single_joint(
            phase,
            angles_L["hip"],
            angles_R["hip"],
            "Hip Angle (Left vs Right)",
            "hip_angles.png",
        )
        self._plot_single_joint(
            phase,
            angles_L["knee"],
            angles_R["knee"],
            "Knee Angle (Left vs Right)",
            "knee_angles.png",
        )
        self._plot_single_joint(
            phase,
            angles_L["ankle"],
            angles_R["ankle"],
            "Ankle Angle (Left vs Right)",
            "ankle_angles.png",
        )

        # --------------------------------------------------------------
        # ROM (left vs right )
        # --------------------------------------------------------------
        self._plot_rom_bars(metrics, filename="rom_comparison.png")

    # ------------------------------------------------------------------
    def _plot_single_joint(self, phase, left, right, title, filename):
        """Plot Left vs Right vs Reference for one joint."""

        plt.figure(figsize=(8, 5))

        plt.plot(phase, left, label="Left", linewidth=2)
        plt.plot(phase, right, label="Right", linewidth=2)

        plt.title(title)
        plt.xlabel("Gait Cycle (%)")
        plt.ylabel("Angle (°)")
        plt.xlim([0, 100])
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.savefig(
            os.path.join(self.person_dir, filename), dpi=300, bbox_inches="tight"
        )
        plt.close()

    # ------------------------------------------------------------------
    def _plot_rom_bars(self, metrics, filename):
        labels = ["Hip", "Knee", "Ankle"]

        L_rom = [
            metrics["left"]["rom_hip"],
            metrics["left"]["rom_knee"],
            metrics["left"]["rom_ankle"],
        ]
        R_rom = [
            metrics["right"]["rom_hip"],
            metrics["right"]["rom_knee"],
            metrics["right"]["rom_ankle"],
        ]

        x = np.arange(len(labels))
        width = 0.35

        plt.figure(figsize=(8, 5))
        plt.bar(x - width / 2, L_rom, width, label="Left")
        plt.bar(x + width / 2, R_rom, width, label="Right")

        plt.xticks(x, labels)
        plt.ylabel("ROM (°)")
        plt.title("Range of Motion (Left vs Right)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.legend()

        out_path = os.path.join(self.person_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    # ------------------------------------------------------------------
