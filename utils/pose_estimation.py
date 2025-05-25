# ---------------------------------------------------
# pose_estimation.txt
# ---------------------------------------------------



# --- Imports ---
import cv2
import mediapipe as mp



# --- Initialize Mediapipe pose detector ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils



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
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
                )
            
        # --- Return Keypoints and Image ---
        return keypoints, image if draw else None