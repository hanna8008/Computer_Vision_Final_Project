# ---------------------------------------------------
# main.py
# ---------------------------------------------------
#
# Main script that loads a person image, extracts body keypoints
# using pose estimation, loads a transparent clothing image, resizes
# it based on shoulder/torso proportions, warps it using keypoints,
# and blends it into the person image. The output image is saved
# in the outputs folder.

# --- Imports ---
from utils.image_utils import load_image, resize_clothing, overlay_image
from utils.pose_estimation import get_body_keypoints
from utils.overlay_clothes import fit_clothing_to_keypoints 
import cv2
import numpy as np

#  Extract Keypoints + Clean Image 
keypoints, person_img = get_body_keypoints("assets/people/test1new.png", draw=False)

# Backup a clean version of the image for neck restoration
_, person_img_clean = get_body_keypoints("assets/people/test1new.png", draw=False)

# Load Transparent Clothing Image 
clothing = load_image("assets/clothes/tshirt_black.png")

# Resize the Clothing to Fit Torso Width/Height 
resized_clothing = resize_clothing(clothing, keypoints[11], keypoints[12], keypoints[23])

# --- Warp the Clothing to Fit Body Keypoints ---
warped_clothing = fit_clothing_to_keypoints(resized_clothing, keypoints, person_img.shape)

# --- Overlay Warped Clothing on the Person Image ---
result = overlay_image(person_img, warped_clothing, (0, 0))

# # --- Restore Neck on Top Using Rectangular Patch ---
# neck_x, neck_y = keypoints[0]

# # Estimate dimensions based on shoulder width
# shoulder_width = abs(keypoints[12][0] - keypoints[11][0])
# neck_w = int(shoulder_width * 0.4)
# neck_h = int(shoulder_width * 0.4)

# # Compute bounding box with clipping
# x1 = max(neck_x - neck_w // 2, 0)
# y1 = max(neck_y - neck_h, 0)
# x2 = min(neck_x + neck_w // 2, result.shape[1])
# y2 = min(neck_y + neck_h, result.shape[0])

# # Copy neck area from clean image
# neck_patch = person_img_clean[y1:y2, x1:x2].copy()
# result[y1:y2, x1:x2] = neck_patch

#  Save Final Output 
cv2.imwrite("outputs/results/tryon_result_tshirt_black_woman1.jpg", result)

#Debug Info 
print("→ Clothing shape (resized):", resized_clothing.shape)
print("→ Clothing shape (warped):", warped_clothing.shape)
# print("→ neck blend at:", (neck_x, neck_y))
print("→ Person image shape:", person_img.shape)