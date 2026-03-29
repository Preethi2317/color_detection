import cv2
from collections import deque

points = deque(maxlen=50)

def update_tracking(cx, cy):
    points.append((cx, cy))

def draw_path(frame, color):
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue
        thickness = 2
        cv2.line(frame, points[i-1], points[i], color, thickness)
