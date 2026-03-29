import cv2
import numpy as np

def preprocess_mask(mask):
    kernel = np.ones((5,5), np.uint8)

    mask = cv2.GaussianBlur(mask, (9,9), 0)
    mask = cv2.medianBlur(mask, 5)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask