
# FitMe: Virtual Clothing Alignment with Pose Estimation

## Project Overview

**FitMe** is a computer vision project that aims to virtually overlay clothing onto real human images using pose estimation.  
The project uses **MediaPipe** to detect anatomical landmarks (e.g., shoulders, hips), and **OpenCV** to resize a transparent shirt image to match the detected person’s proportions.



## Folder Structure

```
.
├── assets/
│   ├── clothes/
│   │   ├── blouse_lightblue.png
│   │   ├── redtank.png
│   │   ├── tank_top_navy.png
│   │   └── tshirt_black.png
│   ├── outputs/
│   │   └── results/
│   │       ├── Blue tshirt blue test.jpg
│   │       ├── keypoints_colored_legend.jpg
│   │       ├── keypoints_overlay.jpg
│   │       ├── keypoints_visualization.jpg
│   │       ├── ladytanktopred.jpg
│   │       ├── Ladytank_keypoints.jpg
│   │       ├── test with tshirt.jpg
│   │       ├── test1new_keypoints.jpg
│   └── people/
│       ├── ladytanklong.jpg
│       ├── mani1_blue.jpg
│       ├── tanktop.webp
│       ├── test1new.png
│       ├── woman1_blouse.jpg
│       ├── woman2_closeup.jpg
│       └── womantanktestimg.jpg
│
├── outputs/
│   └── results/
│       ├── Blue tshirt blue test.jpg
│       ├── keypoints_colored_legend.jpg
│       ├── keypoints_overlay.jpg
│       ├── keypoints_visualization.jpg
│       ├── ladytanktopred.jpg
│       ├── Ladytank_keypoints.jpg
│       ├── tanktop output.jpg
│       ├── tanktop test.jpg
│       ├── test with tank top.jpg
│       ├── test with tshirt.jpg
│       ├── test1new_keypoints.jpg
│       ├── tryon_result.jpg
│       └── tryon_result_tshirt_black_woman1.jpg
│
├── utils/
│   ├── image_utils.py
│   ├── overlay_clothes.py
│   ├── pose_estimation.py
│   └── __pycache__/
│       ├── image_utils.cpython-310.pyc
│       ├── image_utils.cpython-312.pyc
│       ├── overlay_clothes.cpython-310.pyc
│       ├── pose_estimation.cpython-310.pyc
│       ├── pose_estimation.cpython-312.pyc
│       └── pose_estimation.cpython-39.pyc
│
├── LICENSE
├── main.py
├── README.md
├── requirements.txt
└── setup_env.sh
```

---

## Setup Instructions

### 1. Environment Setup

```bash
bash setup_env.sh
conda activate fitme_env
```

> Make sure `conda` is installed and available in your shell.

### 2. Requirements (from `requirements.txt`)

```
opencv-python==4.7.0.72
mediapipe
numpy
gradio
```


---

## Authors

**Hanna Zelis**  
**Asim Wahedna**

---
