import cv2

def create_gui():
    def nothing(x):
        pass

    cv2.namedWindow("Controls")

    cv2.createTrackbar("LH","Controls",0,179,nothing)
    cv2.createTrackbar("LS","Controls",120,255,nothing)
    cv2.createTrackbar("LV","Controls",70,255,nothing)
    cv2.createTrackbar("UH","Controls",10,179,nothing)
    cv2.createTrackbar("US","Controls",255,255,nothing)
    cv2.createTrackbar("UV","Controls",255,255,nothing)

def get_trackbar_values():
    lh = cv2.getTrackbarPos("LH","Controls")
    ls = cv2.getTrackbarPos("LS","Controls")
    lv = cv2.getTrackbarPos("LV","Controls")
    uh = cv2.getTrackbarPos("UH","Controls")
    us = cv2.getTrackbarPos("US","Controls")
    uv = cv2.getTrackbarPos("UV","Controls")

    return (lh, ls, lv), (uh, us, uv)