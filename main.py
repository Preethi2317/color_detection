import cv2
import numpy as np
import time
import os

from detector import detect_colors
from tracker import update_tracking, draw_path
from gui import create_gui, get_trackbar_values

# Folder setup
if not os.path.exists("captures"):
    os.makedirs("captures")

cap = cv2.VideoCapture(0)

create_gui()

colors = {
    "Red": ([0,120,70], [10,255,255], (0,0,255)),
    "Green": ([36,100,100], [86,255,255], (0,255,0)),
    "Blue": ([94,80,2], [126,255,255], (255,0,0))
}

last_save_time = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    # Resize for performance
    frame = cv2.resize(frame, (900, 700))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Slider (optional)
    lower_slider, upper_slider = get_trackbar_values()
    slider_mask = cv2.inRange(hsv, np.array(lower_slider), np.array(upper_slider))
    # Not showing slider window → clean UI

    detections = detect_colors(frame, hsv, colors)

    count_dict = {"Red":0, "Green":0, "Blue":0}

    # Track only ONE object (clean UI)
    max_objects = 3
    count = 0

    for obj in detections:
        if count >= max_objects:
            break

    name = obj["name"]
    x,y,w,h = obj["box"]
    color = obj["color"]

    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 4)
    cv2.putText(frame, name, (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cx = int(x + w/2)
    cy = int(y + h/2)

    cv2.circle(frame, (cx,cy), 5, color, -1)

    # Track ONLY first object to avoid messy lines
    if count == 0:
        update_tracking(cx, cy)
        draw_path(frame, color)

        count_dict[name] += 1
        count += 1

        # Save image every 5 seconds
        if obj["area"] > 3000 and time.time() - last_save_time > 5:
            filename = f"captures/{name}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            last_save_time = time.time()

    # Clean UI panel
    cv2.rectangle(frame, (0,0), (220,120), (0,0,0), -1)

    y_pos = 30
    for k,v in count_dict.items():
        cv2.putText(frame, f"{k}: {v}", (10,y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        y_pos += 30

    # Title
    cv2.putText(frame, "Color Detection System", (200,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Final Output", frame)
    cv2.imshow("Slider Mask", slider_mask)
    cv2.imshow("Slider Mask", cv2.resize(slider_mask, (400,300)))
    cv2.namedWindow("Final Output", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Slider Mask", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)


    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()