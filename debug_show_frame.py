import sys

import cv2

from main import display

cap = cv2.VideoCapture("videos/y5D5FhyDRK4.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 121588)
ret, frame = cap.read()
cap.release()
if not ret:
    print("couldn't get frame")
    sys.exit(1)
display(frame)
