import time
import numpy as np
import cv2
# from mp_detection.pose import PoseDetector
# from mp_detection.face_mesh import FaceMeshDetector
from mp_detection.face import FaceDetector

from utils import *


frame_shape = (640, 480)
width, height = frame_shape

bbox = []
prev = []
detect = False
bow_state = False
xline, yline = 0, 0
xline2, yline2 = 0, 0
gap = 15
times = []
ptime = 0
sec = 0
frame_num = 0

# cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('Ahn.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_shape[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_shape[1])
cap.set(cv2.CAP_PROP_FPS, 30)
fps = cap.get(cv2.CAP_PROP_FPS)
delay = round(1000/fps)

# fourcc = cv2.VideoWriter_fourcc(*'DIVX')
# out = cv2.VideoWriter('bow.avi', fourcc, 30, (frame_shape))

module = FaceDetector(frame_shape)

while cap.isOpened():
    success, image = cap.read()
    image = cv2.resize(image, (frame_shape))
    # image = cv2.flip(image, 1)
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    is_real = module.process(image)
    if is_real:
        bbox = module.get_face_rect()
        
        bbox, prev, detect = low_pass_filter(bbox, prev, detect, 8)
        sx, sy, ex, ey = bbox
        face = image[sy:ey, sx:ex].copy()
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_blur = cv2.GaussianBlur(face_gray, (5, 5), 0.1)

        canny = cv2.Canny(face_blur, 1900, 2000, apertureSize=5, L2gradient=True)
        lines = cv2.HoughLinesP(canny, 1, np.pi/360, 50, minLineLength=50, maxLineGap=10)

        if lines is not None:
            bow = find_line(lines, sx, sy)
            if bow is not None:
                x1, x2, y1, y2 = bow
                prev_line = [xline, yline, xline2, yline2]
                if is_not_move(bow, prev_line, gap):
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    bow_state = True
                    frame_num += 1
                    times.append(frame_num/delay)
                else:
                    xline, yline = x1, y1
                    xline2, yline2 = x2, y2
                    bow_state = False
                    frame_num = 0
                    times.clear() 
        else:
            xline, yline = 0, 0
            xline2, yline2 = 0, 0
            bow_state = False
            frame_num = 0
            times.clear()

        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2, cv2.LINE_AA)

    else:
        pass


    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime
    cv2.putText(image, f'fps:{int(fps)}', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 0, 0), 1)
    
    if len(times) >= 2:
        # sec = times[-1] - times[0]
        sec = times[-1]
    cv2.putText(image, f'{sec:.2f}sec', (width-150, height-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.putText(image, f'bow : {bow_state}', (10, height-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.imshow('Bow detection', image)

    # out.write(image)
    if cv2.waitKey(delay) & 0xFF == 27:
        break
    

# out.release()
cap.release()
cv2.destroyAllWindows()