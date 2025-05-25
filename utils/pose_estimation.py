# ---------------------------------------------------
# pose_estimation.py
# ---------------------------------------------------



# --- Imports ---
import cv2
import mediapipe as mp



# --- Initialize Mediapipe pose detector ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



# --- Define Custom Key Landmarks and Colors ---
KEY_LANDMARKS = {
    # Left shoulder: Blue
    11: ("Left Shoulder", (255, 0, 0)),
    # Right shoulder: Red
    12: ("Right Shoulder", (0, 0, 255)),
    # Left elbow: Orange
    13: ("Left Elbow", (255, 165, 0)),
    # Right elbow: Yellow
    14: ("Right Elbow", (255, 255, 0)),
    # Left Hand: Hot Pink
    15: ("Left Hand", (255, 105, 180)),
    # Right Hand: Medium Purple
    16: ("Right Hand", (147, 112, 219)),
    # Left Hip: Cyan
    23: ("Left Hip", (0, 255, 255)),
    # Right Hip: Green
    24: ("Right Hip", (0, 255, 0)),
    # Neck/Noise: Light Pink
    0: ("Neck/Noise", (255, 182, 193))
}



# --- Get Body Keypoints Function ---
def get_body_keypoints(image_path, draw=False):
    """
    Given a path to a person image, return key body landmarks and optionally draw them.

    Arguments:
        - image_path: str, path to the input image
        - draw: bool, whether to draw landmarks on the image

    Returns:
        - keypoints: dict of {landmark_index: (x, y)} in image coordinates
        - image: original BGR image (with landmarks drawn if draw=True)
    """

    # --- Load and Convert Image ---
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found at path: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    # --- Use Mediapipe Pose in Static Image Mode ---
    with mp_pose.Pose(static_image_mode=True, model_complexity=1, enable_segmentation=False) as pose:
        results = pose.process(image_rgb)

        keypoints = {}

        if results.pose_landmarks:
            for i, lm in enumerate(results.pose_landmarks.landmark):
                x = int(lm.x * width)
                y = int(lm.y * height)
                keypoints[i] = (x, y)

            #Draw Landmarks on image (if draw=True)
            if draw:
                # Draw the full pose skeleton with thicker lines
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4, circle_radius=7),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=4)
                )

                # Draw the custom keypoints with colors
                for idx, (label, color) in KEY_LANDMARKS.items():
                    if idx in keypoints:
                        x, y = keypoints[idx]
                        cv2.circle(image, (x, y), 10, color, -1)
                        cv2.putText(image, label, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1)

                # Add a legend/key to top-left corner
                legend_start_y = 40
                for i, (label, color) in enumerate(KEY_LANDMARKS.values()):
                    y_pos = legend_start_y + i * 30
                    cv2.putText(image, label, (30, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
                    cv2.rectangle(image, (250, y_pos - 20), (280, y_pos + 10), color, -1)
            
        # --- Return Keypoints and Image ---
        return keypoints, image if draw else None