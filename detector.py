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

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            cnt = max(contours, key=cv2.contourArea)

            area = cv2.contourArea(cnt)

            if area > 2000:
                # Smooth bounding box
                x,y,w,h = cv2.boundingRect(cnt)

                cx = int(x + w/2)
                cy = int(y + h/2)

                detections.append({
                    "name": name,
                    "box": (x,y,w,h),
                    "center": (cx, cy),
                    "color": color,
                    "area": area
                })

    return detections