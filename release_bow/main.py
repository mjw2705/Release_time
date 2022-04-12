import time
import numpy as np
import cv2
import mediapipe as mp
import cvlib as cv

from utils import *


num = 0
mode = 'facemesh' if num == 0 else 'facedetect'

frame_shape = (640, 480)
width, height = frame_shape

bbox = []
prev = []
detect = False
bow_state = False
bow_xline, bow_yline = 0, 0
times = []
ptime = 0
sec = 0

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('Ahn.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('bow.avi', fourcc, 30, (frame_shape))

while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (frame_shape))
    # image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    if mode == 'facemesh':
        face_lms = Face_mesh(image)
        if face_lms:
            (sx, sy), (ex, ey) = facemesh_roi(frame_shape, face_lms, 0)
            bbox = [sx, sy, ex, ey]
    else:
        bboxs = Face_detection(image)
        if bboxs:
            sx, sy, ex, ey = facedetect_roi(bboxs, frame_shape)
            bbox = [sx, sy, ex, ey]

    if bbox:
        bbox, prev, detect = low_pass_filter(bbox, prev, detect, 8)
        sx, sy, ex, ey = bbox
        face = image[sy:ey, sx:ex].copy()
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_blur = cv2.GaussianBlur(face_gray, (5, 5), 0.1)
        # release.mp4
        # canny = cv2.Canny(face_blur, 4900, 2000, apertureSize=5, L2gradient=True)
        # lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, minLineLength=56, maxLineGap=16)

        canny = cv2.Canny(face_blur, 1900, 2000, apertureSize=5, L2gradient=True)
        lines = cv2.HoughLinesP(canny, 1, np.pi/360, 50, minLineLength=50, maxLineGap=10)
        cv2.imshow('canny', canny)

        if lines is not None:
            x1, x2, y1, y2 = find_line(lines, sx, sy)
        
            if x1:
                if abs(bow_xline - x1) < 15 or abs(bow_yline - y1) < 15:
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    bow_state = True
                    times.append(time.perf_counter())
                else:
                    bow_xline = x1 
                    bow_yline = y1
                    bow_state = False
                    times.clear()  
        else:
            bow_state = False
            times.clear()

        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2, cv2.LINE_AA)
    else:
        pass

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, f'fps:{int(fps)}', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 0, 0), 1)
    
    if len(times) >= 2:
        sec = times[-1] - times[0]
    cv2.putText(image, f'{sec:.5f}sec', (width-200, height-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.putText(image, f'bow : {bow_state}', (10, height-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.imshow('Bow detection', image)

    out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    

out.release()
cap.release()
cv2.destroyAllWindows()