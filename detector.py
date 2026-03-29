import cv2
import numpy as np
from utils import preprocess_mask

def detect_colors(frame, hsv, colors):
    detections = []

    for name, (lower, upper, color) in colors.items():
        lower = np.array(lower)
        upper = np.array(upper)

        mask = cv2.inRange(hsv, lower, upper)
        mask = preprocess_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)

            if area > 700:
                x,y,w,h = cv2.boundingRect(cnt)

                detections.append({
                    "name": name,
                    "box": (x,y,w,h),
                    "color": color,
                    "area": area
                })

    return detections