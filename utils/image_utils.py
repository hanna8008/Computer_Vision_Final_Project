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
def resize_clothing(clothing_img, shoulder_left, shoulder_right, hip_left):
    """
    Resizes the clothing image to match the shoulder width of the user
    """
    # Calculate the target width based on shoulder distance
    shoulder_width = int(1.1 * np.linalg.norm(np.array(shoulder_right) - np.array(shoulder_left)))
    #shoulder_width = int(0.8 * np.linalg.norm(np.array(shoulder_right) - np.array(shoulder_left)))

    torso_length = int(1.2 * np.linalg.norm(np.array(hip_left) - np.array(shoulder_left)))

    # Preserve aspect ratio
    height, width = clothing_img.shape[:2]
    clothing_aspect = height / width

    new_width = shoulder_width
    new_height = torso_length

    resized = cv2.resize(clothing_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized



# --- Estimate Overlay Position Function ---
def estimate_overlay_position(keypoints, clothing_img, image_shape=None):
    """
    Estimates top-left overlay position (x, y) based on shoulders and hips
    Uses midpoint between shoulders as anchor point
    """
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]

    mid_x = int((left_shoulder[0] + right_shoulder[0]) / 2)
    min_y = int((left_shoulder[1] + right_shoulder[1]) / 2)

    # Estimate torso length (e.g., for long garments)
    torso_length = np.linalg.norm(np.array(keypoints[23]) - np.array(keypoints[11]))  

    # Center the clothing horizontally and place it just below the shoulders
    top_left_x = mid_x - clothing_img.shape[1] // 2
    top_left_y = min_y - int(clothing_img.shape[0] * 0.25)

    # Clamp to image bounds if image_shape provided 
    if image_shape:
        img_h, img_w = image_shape[:2]
        """if top_left_y + clothing_img.shape[0] > img_h:
            clothing_img = clothing_img[:img_h - top_left_y, :, :]"""
        top_left_x = max(0, min(img_w - clothing_img.shape[1], top_left_x))
        top_left_y = max(0, min(img_h - clothing_img.shape[0], top_left_y))
    
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
    alpha_mask = cv2.GaussianBlur(overlay[:, :, 3] / 255.0, (7, 7), 0)


    # Extract region of interest (ROI)
    roi = background[y:y+h, x:x+w]

    # Blend
    for c in range(3):
        roi[:, :, c] = (1.0 - alpha_mask) * roi[:, :, c] + alpha_mask * clothing_rgb[:, :, c]

    background[y:y+h, x:x+w] = roi
    return background