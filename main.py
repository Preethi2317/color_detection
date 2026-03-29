import cv2
import numpy as np
import time
import os


from detector import detect_colors
from tracker import update_tracking, draw_path
from gui import create_gui, get_trackbar_values

# Folder
if not os.path.exists("captures"):
    os.makedirs("captures")

cap = cv2.VideoCapture(0)

create_gui()

colors = {
    "Red": ([0,120,70], [10,255,255], (0,0,255)),
    "Green": ([36,100,100], [86,255,255], (0,255,0)),
    "Blue": ([94,80,2], [126,255,255], (255,0,0))
}

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detections = detect_colors(frame, hsv, colors)

    count_dict = {"Red":0, "Green":0, "Blue":0}

    for obj in detections:
        name = obj["name"]
        x,y,w,h = obj["box"]
        color = obj["color"]

        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, name, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cx = int(x + w/2)
        cy = int(y + h/2)

        cv2.circle(frame, (cx,cy), 5, color, -1)

        update_tracking(cx, cy)
        draw_path(frame, color)

        count_dict[name] += 1

        if obj["area"] > 2000:
            filename = f"captures/{name}_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)

    # Clean UI panel
    y_pos = 30
    for k,v in count_dict.items():
        cv2.putText(frame, f"{k}: {v}", (10,y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        y_pos += 30

    cv2.imshow("Final Output", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()