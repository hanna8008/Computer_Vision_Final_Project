# ---------------------------------------------------
# main.py
# ---------------------------------------------------
#
# Main script that loads a person image, extracts body keypoints
# using pose estimation, loads a transparent clothing image, resizes
# it based on shoulder/torso proportions, estimates the overlay
# position, and blends it into the person image. The output image
# is saved in the outputs folder.



# --- Imports ---
from utils.image_utils import load_image, resize_clothing, overlay_image, estimate_overlay_position
from utils.pose_estimation import get_body_keypoints
import cv2



# --- Extract Pose Keypoints from Person Image ---
# Load the image and extract landmrks (shoulders, hips, etc.)
keypoints, person_img = get_body_keypoints("assets/people/woman1_blouse.jpg", draw=True)



# --- Load Clothing Image ---
# Load a transparent PNG shirt
clothing = load_image("assets/clothes/tshirt_black.png")



# --- Resize Clothing to Fit Body Dimensions ---
# Resize the shirt to match shoulder width and torso height 
resized_clothing = resize_clothing(clothing, keypoints[11], keypoints[12], keypoints[23])



# --- Estimate Placement Position on Person ---
# Determine the top-left corner where the shirt should be placed
position = estimate_overlay_position(keypoints, resized_clothing, person_img.shape)



# --- Overlay Clothig onto the Person Image ---
# Blend the shirt image onto the person using the alpha channel
result = overlay_image(person_img, resized_clothing, position)



# --- Save Final Output Image ---
# Save the composite result
cv2.imwrite("outputs/results/tryon_result_tshirt_black_woman1.jpg", result)



# --- Debug Info ---
# Print key debugging details to verify correctness
print("→ Position:", position)
print("→ Clothing shape:", resized_clothing.shape)
print("→ Person shape:", person_img.shape)