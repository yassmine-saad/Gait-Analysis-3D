# 🏃‍♂️ 3D Markerless Gait Analysis & Autonomous Following Robot



A high-precision biomechanical analysis system that follows a subject and extracts clinical gait metrics using 3D skeletal reconstruction and depth sensing.



Developed by engineering students at **ENISO** (National School of Engineers of Sousse).



🎓 **Project by:** [Yahya Ben Turkia](https://github.com/yahya-bt), [Yasmine Saad](https://github.com/yassmine-saad)



🧑‍🏫 Supervised by: Dr. Lamine Houssein — PhD in Robotics, Assistant Professor at ENISO



---



## 🎯 Objective



Design and implement a system capable of:

- **Markerless Detection:** Identifying a person using **YOLOv8** and tracking 133 keypoints via **MMPose (HRNet)**.

- **3D Reconstruction:** Converting 2D vision data into 3D coordinates using **Intel RealSense D435i** depth alignment.

- **Biomechanical Analysis:** Transforming data into the **Sagittal Plane** to calculate Hip, Knee, and Ankle angles.

- **Signal Integrity:** Applying **Kalman Filtering** and **RTS Smoothing** for clinical-grade data.

- **Autonomous Following:** Maintaining a **safe distance (3.0m)** via **Modbus TCP** commands.



---



## 🛠️ Technologies Used



- 📷 **Intel RealSense D435i** — RGB + Depth Camera

- 🧍‍♂️ **MMPose & YOLOv8** — Advanced 2D/3D Pose Estimation

- 🧠 **Python 3.10** with:

&nbsp; - `ultralytics` (YOLOv8)

&nbsp; - `mmpose` (Skeletal Tracking)

&nbsp; - `filterpy` (Kalman/RTS Smoothing)

&nbsp; - `pyrealsense2` (Camera API)

&nbsp; - `pyModbusTCP` (Robot Control)

&nbsp; - `fpdf` (Clinical Report Generation)

- ⚙️ **Modbus TCP** — Industrial protocol for robot communication



---



## 🏗️ Project Structure



```text

├── camera/
│   ├── __init__.py
│   └── camera.py                # RealSense alignment & acquisition

├── vision/
│   ├── __init__.py
│   ├── pose_estimator.py        # 3D reconstruction & RTS Smoothing
│   └── GaitAnalyzer.py          # Biomechanical math & segmentation

├── robot/
│   ├── __init__.py
│   └── follow_controller.py     # Modbus-based PID distance control

├── utils/
│   ├── __init__.py
│   └── utils.py                 # YOLOv8 Person detection logic

├── visualisation/
│   ├── __init__.py
│   └── visualizer.py            # Gait curve & ROM plotting

├── exporter/
│   ├── __init__.py
│   └── exporter.py              # PDF Report & CSV generation

├── main.py                      # Main supervisor script

├── environment.yml              # Conda environment setup

└── requirements.txt             # Pip dependencies

```

### 🚀 Installation & Environment Setup (Reproducible)
#### ⚠️ Important Note

To guarantee identical behavior across machines, the development   Conda environment was packaged using conda-pack.
This avoids Conda/Pip conflicts and CUDA-related reinstallation issues.

#### ✅ System Requirements

* Operating System: Windows

* GPU: NVIDIA GPU

* CUDA: Same CUDA version as installed on the development machine

* Conda: Miniconda or Anaconda already installed

#### 📦 Files 

* **gait_env_new.tar.gz** → The packaged Conda environment (`gait_env_new.tar.gz`) is **too large for GitHub**, so please download it from the link below:

**Download link:** [Google Drive - gait_env_new.tar.gz](https://drive.google.com/file/d/1YZhldUfOMBrwaKC3vU4lPZU70cwi67cf/view?usp=sharing)

* **Project source code** (this repository)

#### 1. Locate the Conda environments directory

Open Anaconda Prompt and run:
```
conda env list

```
You will see output similar to:
```
base    *  C:\Users\USERNAME\miniconda3
env1       C:\Users\USERNAME\miniconda3\envs\env1
```
The Conda environments directory is:
```
C:\Users\USERNAME\miniconda3\envs
```

#### 2. Create the environment folder:
Replace the path with the one found in Step 1:

```
mkdir C:\Users\USERNAME\miniconda3\envs\gait_env_new

```
#### 3️. Extract the packaged environment

From the directory containing gait_env_new.tar.gz:
```
tar -xzf gait_env_new.tar.gz -C  C:\Users\USERNAME\miniconda3\envs\gait_env_new
```
#### 4️. Fix internal paths (MANDATORY)

Run once after extraction:
```
C:\Users\USERNAME\miniconda3\envs\gait_env_new\Scripts\conda-unpack.exe
```

⚠️ This step adapts absolute paths to the new machine and is required.

#### 5️. Activate the environment
```
conda activate gait_env_new
```
#### 6️. Verify the environment
```
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"

```
Expected:

* Python 3.10.x

* PyTorch with CUDA support

* True for CUDA availability
### 🎮 How to Use

⚠️ **Prerequisite:** EduBot Connection

Ensure the EduBot is powered on and connected to your pc via usb cable.



**Robot IP:** Ensure the IP in robot/follow_controller.py matches the EduBot's Modbus server address.



**Camera:** Connect the Intel RealSense D435i via USB 3.0.



**Execution Steps:**



#### 1. Launch System:

```

python main.py

```

#### 2. Recording Logic:



'r' (Start): Begins recording ONLY if a person is detected by the vision system.



's' (Stop): Ends recording and immediately triggers offline analysis.



#### 3. View Results: 
A new folder will be created in curves/ containing your PDF Report, gait graphs, and CSV data.



### 📊 Methodology

The system follows a clinical workflow:



1. Pose Projection: Mapping 2D points to 3D space using the Pinhole Camera model.



2. Sagittal Alignment: Rotating the 3D skeleton to align with the walking direction.



3. Filtering: Using the Rauch-Tung-Striebel (RTS) smoother to remove depth noise.



4. Gait Normalization: Segmenting steps into a 0-100% phase for standard clinical comparison.



### 🎓 Academic Context

**Institution:** National School of Engineers of Sousse (ENISO)



**Major:** Mechatronics Engineering (Méca 3.1)



**Project Type:** Semester Project 2025-2026










