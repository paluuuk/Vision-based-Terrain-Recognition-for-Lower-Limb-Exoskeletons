# Exoskeleton Terrain Perception

Real-time terrain perception pipeline for a lower-limb exoskeleton using stereo depth sensing, 3D point-cloud processing, and CNN-based classification.

The system classifies **7 terrain categories** — level ground, ascending stairs, descending stairs, up slope, down slope, obstacle, and gap — and is designed to support terrain-aware gait adaptation in robotic mobility systems.

## Why this project matters

Lower-limb exoskeletons need reliable perception to switch locomotion modes safely. Outdoor environments make this difficult: sunlight, reflective surfaces, missing depth values, camera motion, and noisy point clouds can degrade terrain recognition.

This project explores a more robust perception stack built around **ZED stereo depth**, depth-map enhancement, point-cloud normalization, compact terrain representations, and CNN inference.

## System Pipeline

```text
ZED Stereo Camera
        ↓
Depth Acquisition
        ↓
Confidence Filtering / Neural Depth
        ↓
ROI Extraction
        ↓
Depth Inpainting + Denoising
        ↓
Temporal Filtering
        ↓
3D Point-Cloud Reconstruction
        ↓
Orientation Compensation
        ↓
Spatial Filtering / Downsampling
        ↓
100 × 100 Binary Terrain Representation
        ↓
CNN Inference
        ↓
Terrain Class + Estimated Geometry
```

## Engineering Highlights

### Stereo depth and camera integration

The pipeline integrates the **Stereolabs ZED SDK** for stereo depth capture, camera intrinsics, positional tracking, and IMU orientation data. Earlier experiments were based on Intel RealSense hardware; ZED-based variants were explored to improve robustness under outdoor lighting and reflective surfaces.

### Depth-map robustness

Multiple preprocessing strategies are implemented across the repository:

- confidence and texture-confidence filtering
- ZED neural depth mode
- region-of-interest extraction
- Navier–Stokes depth inpainting with OpenCV
- non-local means denoising
- temporal averaging across consecutive depth frames

These steps target missing or unstable depth measurements before 3D reconstruction.

### Point-cloud processing

Depth pixels are converted into 3D coordinates using camera intrinsic parameters. The pipeline then:

- removes invalid / NaN samples
- downsamples the point cloud
- compensates for camera orientation using IMU-derived rotation
- constrains the lateral region used for classification
- normalizes terrain height and distance
- converts the filtered point cloud into a compact **100 × 100 binary representation**

### CNN terrain classifier

The classifier is implemented in TensorFlow/Keras using stacked convolutional blocks with:

- Conv2D layers
- batch normalization
- max pooling
- dropout
- L2 regularization
- fully connected classification layers
- softmax output across seven terrain classes

The project work achieved approximately **95% terrain-classification accuracy** across seven classes during development and evaluation.

## Repository Contents

The current repository contains several experiment and integration variants developed while improving the perception pipeline:

| File | Purpose |
|---|---|
| `Terrain_Recognition_ZED_ROI.py` | ZED-based terrain-recognition pipeline with ROI processing |
| `Terrain_Recognition_ZED_inpaint.py` | Depth inpainting, denoising, temporal filtering, point-cloud processing, and CNN inference |
| `Terrain_Recognition_ZED_neural.py` | Experiments using ZED neural depth mode |
| `batch_run_DL_client.py` | Batch/deep-learning inference workflow |
| `batch_run_DL_client_inpainting.py` | Batch workflow incorporating depth inpainting |

## Tech Stack

**Languages / ML**  
Python · TensorFlow · Keras · NumPy · SciPy

**Computer Vision / Robotics**  
OpenCV · ZED SDK · stereo depth · 3D point clouds · IMU orientation · ROS integration context

**Hardware / Deployment Context**  
Stereolabs ZED 2/2i · Intel RealSense D435i · NVIDIA Jetson Nano

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU recommended for ZED / CNN workflows
- OpenCV
- TensorFlow / Keras
- ZED SDK for ZED-based pipelines
- Intel RealSense SDK for RealSense variants

### Setup

```bash
git clone https://github.com/paluuuk/Vision-based-Terrain-Recognition-for-Lower-Limb-Exoskeletons.git
cd Vision-based-Terrain-Recognition-for-Lower-Limb-Exoskeletons

python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate      # Windows
```

Install the Python dependencies required by the specific experiment you want to run, together with the appropriate camera SDK.

> Some scripts expect trained model weights such as `best_model.h5` and compatible ZED hardware. These artifacts are not currently included in the repository.

## Current Repository Status

This repository reflects the research and experimentation phase of the project rather than a production software package. Several variants contain overlapping camera, preprocessing, model, and inference logic.

A natural next engineering step is to separate the pipeline into reusable modules for camera I/O, preprocessing, point-cloud processing, model definition, and inference while adding reproducible dependency management and automated tests.

## Research Context

This work was developed as part of a **Nanyang Technological University final-year project** in collaboration with the **Rehabilitation Research Institute of Singapore (RRIS)**, focused on vision-based terrain recognition for lower-limb exoskeleton control.

The broader goal is reliable perception under real-world conditions where depth cameras can struggle with glare, reflective materials, changing illumination, and missing depth measurements.