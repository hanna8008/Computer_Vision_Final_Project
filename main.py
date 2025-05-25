# ---------------------------------------------------
# main.py
# ---------------------------------------------------



# --- Imports ---
from utils.pose_estimation import get_body_keypoints
import cv2


# Use one of your person images
image_path = "assets/people/woman1_blouse.jpg"


# Run pose estimation with landmark drawing enabled
keypoints, image = get_body_keypoints(image_path, draw=True)


# Save the output image
cv2.imwrite("outputs/results/keypoints_colored_legend.jpg", image)

# Optional: print key landmarks
print("Left shoulder:", keypoints.get(11))
print("Right shoulder:", keypoints.get(12))
