# Vision-based Terrain Recognition for Lower-Limb Exoskeletons (Outdoor)

A vision-based terrain recognition pipeline designed for a lower-limb exoskeleton to classify outdoor walking environments in real time. The system processes depth-derived 3D point clouds, converts them into compact binary representations, and uses a CNN-based classifier to recognize **7 terrain classes**:

**[Level Ground, Ascending Stairs, Descending Stairs, Up Slope, Down Slope, Obstacle, Gap]** 

This project was developed as part of an NTU Final Year Project focused on making exoskeleton terrain perception more robust under real-world outdoor conditions (glare, reflections, missing depth, mixed lighting).

---

## Why this matters

Lower-limb exoskeletons need reliable perception to safely switch locomotion modes (stairs vs slopes vs obstacles). Vision systems can fail outdoors due to sunlight, reflective materials (metal/glass), and depth-map holes leading to misclassification and safety risk.

---

## System overview

### Core pipeline (high level)
1. **Depth capture → 3D point cloud**
2. **Point cloud pre-processing + dimensionality reduction**
3. **Binary image generation** (compact representation of terrain geometry)
4. **CNN terrain classification**
5. Output: **terrain class + estimated distance**, passed to the exoskeleton controller for gait adaptation 

### Hardware / integration context
- Initially tested with **Intel RealSense D435i** 
- Improved outdoor robustness by transitioning to **Stereolabs ZED (2/2i)** (polarizer helps reduce glare/reflections) 
- Exoskeleton integration context includes **Jetson Nano** and **ROS** for module integration (as used in the underlying exoskeleton stack)

---

## Key improvements explored in this work

### 1) ZED camera integration (vs RealSense)
ZED SDK setup and integration required adapting the pipeline to ZED APIs/data formats, then validating depth/point cloud streams using sample apps. 

### 2) ZED Neural Depth / confidence improvements
Explores improving depth quality under challenging lighting/reflective surfaces (see thesis sections on neural depth mapping).

### 3) Region of Interest (ROI) on depth map
Restricts processing to the most relevant portion of the depth map to stabilize features and improve classification consistency (especially outdoors).

### 4) Depth inpainting to fill missing depth
Uses **Navier–Stokes inpainting** to fill holes in depth maps and reduce classification failures caused by missing/garbled depth.

---

## Results / observations (high level)

- Base RealSense pipeline performs well indoors on simple obstacles, but struggles with reflective objects and mixed lighting (accuracy drops significantly in those cases). 
- Outdoor lighting can cause slope/stairs confusion due to depth/point-cloud distortions. 
- ZED (with polarizer + depth enhancements) is explored specifically to improve performance in these outdoor failure modes.   

---

## Getting started


### Prerequisites
- Python 3.8+ (recommended)
- CUDA-capable GPU (recommended for ZED / CNN inference)
- OpenCV
- For **RealSense**: Intel RealSense SDK (librealsense)
- For **ZED**: ZED SDK + CUDA + OpenCV configured correctly 

### Install (example)
```bash
git clone https://github.com/paluuuk/Vision-based-Terrain-Recognition-for-Lower-Limb-Exoskeletons.git
cd Vision-based-Terrain-Recognition-for-Lower-Limb-Exoskeletons

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
