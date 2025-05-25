from utils.image_utils import load_image, resize_clothing, overlay_image, estimate_overlay_position
from utils.pose_estimation import get_body_keypoints
import cv2

# Load person and get keypoints
keypoints, person_img = get_body_keypoints("assets/people/woman1_blouse.jpg", draw=True)

# Load clothing PNG
clothing = load_image("assets/clothes/tshirt_black.png")

# Resize clothing to match shoulder width
resized_clothing = resize_clothing(clothing, keypoints[11], keypoints[12], keypoints[23])

# Estimate where to place the clothing
position = estimate_overlay_position(keypoints, resized_clothing, person_img.shape)

"""#Draw bounding box where overlay will go
cv2.rectangle(
    person_img,
    position,
    (position[0] + resized_clothing.shape[1], position[1] + resized_clothing.shape[0]),
    (0, 255, 255),  # yellow-green color
    4  # thickness
)"""

# Overlay the shirt onto the person image
result = overlay_image(person_img, resized_clothing, position)

cv2.imwrite("outputs/results/tryon_result_tshirt_black_woman1.jpg", result)

print("→ Position:", position)
print("→ Clothing shape:", resized_clothing.shape)
print("→ Person shape:", person_img.shape)