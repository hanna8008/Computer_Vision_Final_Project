
# FitMe: Virtual Clothing Alignment with Pose Estimation

## Project Overview

**FitMe** is a two-part computer vision project that aims to virtually overlay clothing onto real human images using pose estimation.  
This repository contains **Phase 1**, completed by **Hanna**, which focuses on detecting upper-body keypoints and resizing clothing images accordingly.

The project uses **MediaPipe** to detect anatomical landmarks (e.g., shoulders, hips), and **OpenCV** to resize a transparent shirt image to match the detected person’s proportions.

---

## Phase 1 Responsibilities 

This half of the project includes:
- Creating full file/folder structure
- Building the Conda environment and setup script
- Extracting pose keypoints (shoulders, hips, neck)
- Resizing a transparent shirt PNG to match the person’s width and height
- Saving visualizations of resized clothing and detected keypoints

> Note: Final clothing overlay realism (arm occlusion, body contouring, full integration) is **not included** in Phase 1. That is part of Phase 2.

---

## Folder Structure

```
FitMe/
├── assets/
│   ├── clothes/
│   │   ├── blouse_lightblue.png
│   │   ├── dress_red_flowy.png
│   │   ├── hoodie_maroon.png
│   │   ├── jacket_black.png
│   │   ├── tank_top_navy.png
│   │   └── tshirt_black.png
│   └── people/
│       ├── man1_blue.jpg
│       ├── woman1_blouse.jpg
│       └── woman2_closeup.jpg
├── gui/
│   └── app.py
├── outputs/
│   └── results/
│       ├── keypoints_colored_legend.jpg
│       ├── keypoints_overlay.jpg
│       ├── tryon_result.jpg
│       └── tryon_result_tshirt_black_woman1.jpg
├── utils/
│   ├── image_utils.py
│   ├── overlay_clothes.py
│   └── pose_estimation.py
├── main.py
├── requirements.txt
├── run_gui.sh
├── setup_env.sh
└── README.md
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

## How to Run

```bash
python main.py
```

This script:
- Loads a test image from `assets/people/`
- Detects pose keypoints using MediaPipe
- Loads a transparent PNG shirt from `assets/clothes/`
- Resizes the shirt based on shoulder width and torso length
- Saves the resized clothing and keypoints overlay to `outputs/results/`

---

## Sample Outputs

- `keypoints_overlay.jpg` — shows detected pose landmarks
- `tryon_result_tshirt_black_woman1.jpg` — shows resized clothing image
- `keypoints_colored_legend.jpg` — visual guide to landmark indices

> Imperfect fit and floaty overlays are expected in this phase! Final rendering logic will be handled by Phase 2.

---

## Handoff Notes for Partner (Phase 2)

You’ll pick up where this phase ends:

**Your responsibilities may include:**
- Accurate overlay of shirt onto torso
- Handling arm occlusion and sleeve alignment
- Smoothing fit using segmentation or keypoint interpolation
- Supporting dynamic garments or accessory layering

---

## Authors

**Hanna Zelis**  
**Asim Wuhadna**

---