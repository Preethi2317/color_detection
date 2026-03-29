import streamlit as st
import cv2
import numpy as np
from collections import deque

st.set_page_config(layout="wide")

st.title("🎨 Color Detection System")

# Sidebar
mode = st.sidebar.selectbox(
    "Select Mode",
    ["Camera Detection", "HSV Slider", "About"]
)

# Tracking buffer
points = deque(maxlen=20)

def preprocess(mask):
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.GaussianBlur(mask, (9,9), 0)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

# Color ranges
colors = {
    "Red": ([0,120,70], [10,255,255], (0,0,255)),
    "Green": ([36,100,100], [86,255,255], (0,255,0)),
    "Blue": ([94,80,2], [126,255,255], (255,0,0))
}

cap = cv2.VideoCapture(0)

# -----------------------------
# CAMERA DETECTION MODE
# -----------------------------
if mode == "Camera Detection":

    st.subheader("Live Detection")

    col1, col2 = st.columns([3,1])

    frame_display = col1.empty()
    info_display = col2.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800,600))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        count = {"Red":0, "Green":0, "Blue":0}

        main_center = None

        for name, (lower, upper, color) in colors.items():
            lower = np.array(lower)
            upper = np.array(upper)

            mask = cv2.inRange(hsv, lower, upper)
            mask = preprocess(mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)

                if area > 2500:
                    x,y,w,h = cv2.boundingRect(cnt)

                    # Draw stable rectangle
                    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 3)
                    cv2.putText(frame, name, (x,y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    cx = int(x + w/2)
                    cy = int(y + h/2)

                    cv2.circle(frame, (cx,cy), 6, color, -1)

                    count[name] += 1

                    # Track only one object
                    if main_center is None:
                        main_center = (cx, cy)

        # Smooth tracking (no zig-zag)
        if main_center:
            points.appendleft(main_center)

        for i in range(1, len(points)):
            if points[i-1] is None or points[i] is None:
                continue

            thickness = int(4 * (1 - i/20)) + 1
            cv2.line(frame, points[i-1], points[i], (0,255,255), thickness)

        # Display frame
        frame_display.image(frame, channels="BGR")

        # Display counts
        info_display.markdown(f"""
        ### Object Count
        🔴 Red: {count["Red"]}  
        🟢 Green: {count["Green"]}  
        🔵 Blue: {count["Blue"]}
        """)

# -----------------------------
# HSV SLIDER MODE
# -----------------------------
elif mode == "HSV Slider":

    st.subheader("HSV Tuning")

    lh = st.slider("LH", 0,179,0)
    ls = st.slider("LS", 0,255,120)
    lv = st.slider("LV", 0,255,70)
    uh = st.slider("UH", 0,179,10)
    us = st.slider("US", 0,255,255)
    uv = st.slider("UV", 0,255,255)

    frame_display = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (800,600))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([lh,ls,lv])
        upper = np.array([uh,us,uv])

        mask = cv2.inRange(hsv, lower, upper)
        mask = preprocess(mask)

        frame_display.image(mask, channels="GRAY")

# -----------------------------
# ABOUT
# -----------------------------
else:
    st.subheader("About Project")

    st.write("""
    This project performs real-time color detection using computer vision.

    Features:
    - Multi-color detection (Red, Green, Blue)
    - Noise reduction using preprocessing
    - Smooth object tracking (no zig-zag)
    - HSV slider tuning
    - Clean single-window UI
    """)