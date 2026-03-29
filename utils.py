import cv2
import numpy as np
def preprocess_mask(mask):
    kernel = np.ones((5,5), np.uint8)
    
    mask = cv2.GaussianBlur(mask, (5,5), 0)  # smooth noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)

    return mask