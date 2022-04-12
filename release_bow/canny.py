import time
import datetime
import numpy as np
import cv2
import mediapipe as mp


def nothing(x):
    pass

frame_shape = (640, 480)

# cap = cv2.VideoCapture('lee.mp4')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 30)

cv2.namedWindow('canny')
cv2.createTrackbar('threshold1', 'canny', 1900, 3000, nothing)
cv2.createTrackbar('threshold2', 'canny', 1900, 3000, nothing)

while cap.isOpened():

    success, image = cap.read()
    image = cv2.resize(image, (frame_shape))
    # image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    dst = image.copy()
    # face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

    face_blur = cv2.GaussianBlur(dst, (5, 5), 0.1)
    threshold1 = cv2.getTrackbarPos('threshold1', 'canny')
    threshold2 = cv2.getTrackbarPos('threshold2', 'canny')
    
    canny = cv2.Canny(face_blur, threshold1, threshold2, apertureSize=5, L2gradient=True)

    cv2.imshow('canny', canny)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cv2.destroyAllWindows()