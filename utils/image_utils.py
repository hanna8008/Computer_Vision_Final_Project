# ---------------------------------------------------
# image_utils.py
# ---------------------------------------------------



# --- Imports ---
import cv2
import numpy as np



# --- Load Image Function ---
def load_image(image_path):
    """
    Load an image from the specified path
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image



# --- Resize Clothing Function ---
def resize_clothing(clothing_img, shoulder_left, shoulder_right):
    """
    Resizes the clothing image to match the shoulder width of the user
    """
    # Calculate the target width based on shoulder distance
    shoulder_width = int(np.linalg.norm(np.array(shoulder_right) - np.array(shoulder_left)))

    # Preserve aspect ratio
    height, width = clothing_img.shape[:2]
    aspect_ratio = height / width
    new_height = int(shoulder_width * aspect_ratio)

    resized = cv2.resize(clothing_img, (shoulder_width, new_height), interpolation=cv2.INTER_AREA)
    return resized



# --- Estimate Overlay Position Function ---
def estimate_overlay_position(keypoints, clothing_img):
    """
    Estimates top-left overlay position (x, y) based on shoulders and hips
    Uses midpoint between shoulders as anchor point
    """
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]

    mid_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
    min_y = int((left_shoulder[1] + right_shoulder[1]))

    # Center the clothing horizontally and place it just below the shoulders
    top_left_x = int(mid_x - clothing_img.shape[1] / 2)
    top_left_y = min_y
    
    return (top_left_x, top_left_y)



# --- Overlay Image Function ---
def overlay_image(background, overlay, position):
    """
    Overlays a transparant image (e.g. shirt) onto a background (person)
    """
    x, y = position
    h, w = overlay.shape[:2]

    # Ensure overlay fits in the background image
    if x < 0 or y < 0 or x + w > background.shape[1] or y + h > background.shape[0]:
        print("Overlay is out of bounds. Skipping")
        return background
    
    # Split out the alpha mask
    clothing_rgb = overlay[:, :, :3]
    alpha_mask = overlay[:, :, 3] / 255.0

    # Extract region of interest (ROI)
    roi = background[y:y+h, x:x+w]

    # Blend
    for c in range(3):
        roi[:, :, c] = (1.0 - alpha_mask) * roi[:, :, c] + alpha_mask * clothing_rgb[:, :, c]

    background[y:y+h, x:x+w] = roi
    return background