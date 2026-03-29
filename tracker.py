import cv2
from collections import deque

points = deque(maxlen=20)

def update_tracking(cx, cy):
    points.appendleft((cx, cy))   # important change

def draw_path(frame, color):
    for i in range(1, len(points)):
        if points[i-1] is None or points[i] is None:
            continue

        thickness = int(3 * (1 - i/20)) + 1
        cv2.line(frame, points[i-1], points[i], color, thickness)
