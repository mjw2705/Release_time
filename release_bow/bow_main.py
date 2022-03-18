import time
import datetime
import numpy as np
import cv2
import mediapipe as mp

from utils import *


num = 1
mode = 'facemesh' if num == 0 else 'facedetect'

frame_shape = (740, 580)
bboxs = []
raw_state = False
times = []
bow_xline, bow_yline = 0, 0
ptime = 0
sec = 0

# cap = cv2.VideoCapture(1, cv2.CAP_DS HOW)
cap = cv2.VideoCapture('lee.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('bow_detect.avi', fourcc, 30, (frame_shape))

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
            bbox = (sx, sy, ex, ey)
            # # 입 roi
            # mouse_lms = [face_lms[n_idx] for n_idx in mouse_idxs]
            # abs_mouse_lms = Convert_abs_lms(frame_shape, mouse_lms)
            # (m_sx, m_sy), (m_ex, m_ey) = mouse_roi(frame_shape, abs_mouse_lms, 20)
            # face = image[m_sy:m_ey, m_sx:m_ex].copy()
    else:
        bboxs = Face_detection(image)
        if bboxs:
            sx, sy, ex, ey = facedetect_roi(bboxs, frame_shape)
            bbox = (sx, sy, ex, ey)

    if bbox:
        face = image[sy:ey, sx:ex].copy()
        face_h, face_w = face.shape[:2]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_blur = cv2.GaussianBlur(face_gray, (5, 5), 0.1)
        # release.mp4
        # canny = cv2.Canny(face_blur, 4900, 2000, apertureSize=5, L2gradient=True)
        # lee.mp4
        canny = cv2.Canny(face_blur, 3000, 2000, apertureSize=5, L2gradient=True)
        # canny = cv2.Canny(face_blur, 1900, 2000, apertureSize=5, L2gradient=True)
        cv2.imshow('canny', canny)
        lines = cv2.HoughLinesP(canny, 1, np.pi/360, 50, minLineLength=50, maxLineGap=10)
        # lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, minLineLength=56, maxLineGap=16)
        # lines = cv2.HoughLinesP(canny, 1, np.pi/180, 50, minLineLength=50, maxLineGap=20)

        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    slope = (y2 - y1) // (x2 - x1)
                    if -20 < slope < -2:
                        print(line)
                        x1 = int(x1 + sx)
                        x2 = int(x2 + sx)
                        y1 = int(y1 + sy)
                        y2 = int(y2 + sy)
                        # x1 = int(x1 + m_sx)
                        # x2 = int(x2 + m_sx)
                        # y1 = int(y1 + m_sy)
                        # y2 = int(y2 + m_sy)
                        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            
            if abs(bow_xline - x1) < 10 or abs(bow_yline - y1) < 10:
                raw_state = True
                times.append(time.perf_counter())
            else:
                bow_xline = x1 
                bow_yline = y1
                raw_state = False
                times.clear()  
        else:
            raw_state = False
            times.clear()
        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2, cv2.LINE_AA)
        # cv2.rectangle(image, (m_sx, m_sy), (m_ex, m_ey), (0, 0, 255), 2, cv2.LINE_AA)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, f'fps:{int(fps)}', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 0, 0), 1)
    
    if len(times) >= 2:
        sec = times[-1] - times[0]
    cv2.putText(image, f'{sec:.5f}sec', (frame_shape[0]-200, frame_shape[1]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.putText(image, f'bow : {raw_state}', (10, frame_shape[1]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.imshow('Bow detection', image)

    out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    

out.release()
cap.release()
cv2.destroyAllWindows()