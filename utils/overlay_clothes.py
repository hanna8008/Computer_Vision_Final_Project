import cv2
import numpy as np
def scale_point(p, center, factor):
    return (
        int(center[0] + factor * (p[0] - center[0])),
        int(center[1] + factor * (p[1] - center[1]))
    )

def fit_clothing_to_keypoints(clothing_img, keypoints, person_shape):
    h, w = clothing_img.shape[:2]
    img_h, img_w = person_shape[:2]

    # Shirt image reference points
    src_pts = np.float32([
        [0, 0],
        [w - 1, 0],
        [w // 2, h - 1]
    ])

    # Body triangle
    left_shoulder = keypoints[11]
    right_shoulder = keypoints[12]
    mid_hip = (
        (keypoints[23][0] + keypoints[24][0]) // 2,
        (keypoints[23][1] + keypoints[24][1]) // 2
    )

    # Center point to scale around
    center = (
        (left_shoulder[0] + right_shoulder[0] + mid_hip[0]) // 3,
        (left_shoulder[1] + right_shoulder[1] + mid_hip[1]) // 3
    )

    # Amplify the triangle by scaling outwards from center
    scale_factor = 1.6  
    dst_pts = np.float32([
        scale_point(left_shoulder, center, scale_factor),
        scale_point(right_shoulder, center, scale_factor),
        scale_point(mid_hip, center, scale_factor)
    ])

    # Affine warp
    M = cv2.getAffineTransform(src_pts, dst_pts)
    warped = cv2.warpAffine(clothing_img, M, (img_w, img_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    
    return warped