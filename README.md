
# FitMe: Virtual Clothing Alignment with Pose Estimation

## Project Overview

**FitMe** is a two-part computer vision project that aims to virtually overlay clothing onto real human images using pose estimation.  
This repository contains **Phase 1**, completed by **Hanna**, which focuses on detecting upper-body keypoints and resizing clothing images accordingly.

The project uses **MediaPipe** to detect anatomical landmarks (e.g., shoulders, hips), and **OpenCV** to resize a transparent shirt image to match the detected person’s proportions.



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

## Authors

**Hanna Zelis**  
**Asim Wahedna**

---
