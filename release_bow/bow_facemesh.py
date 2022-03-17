import time
import datetime
import numpy as np
import cv2
import mediapipe as mp

def face_roi(frame_shape, face_lms, margin):
    sx, sy = frame_shape
    ex, ey = 0, 0
    for lms in face_lms:
        cx, cy = int(lms[0] * frame_shape[0]), int(lms[1] * frame_shape[1])
        if cx < sx: sx = cx
        if cy < sy: sy = cy
        if cx > ex: ex = cx
        if cy > ey: ey = cy
    
    return (sx - margin, sy - margin), (ex + margin, ey + margin)

def Face_mesh(image):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,
                                                max_num_faces=1,
                                                min_detection_confidence=0.7)
    image.flags.writeable = False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    image.flags.writeable = True

    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            face_lms = [[lms.x, lms.y, lms.z] for lms in landmarks.landmark] #len 468
        return face_lms
    return None

def Convert_abs_lms(frame_shape, lms):
    abs_lms = np.array(lms)
    abs_lms[:, 0] = abs_lms[:, 0] * frame_shape[0]
    abs_lms[:, 1] = abs_lms[:, 1] * frame_shape[1]
    return abs_lms[:, :2]

def Draw_landmark(image, lms):
    for lm in lms:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)

def mouse_roi(frame_shape, lms, margin):
    sx, sy = frame_shape
    ex, ey = 0, 0
    for lm in lms:
        cx, cy = int(lm[0]), int(lm[1])
        if cx < sx: sx = cx
        if cy < sy: sy = cy
        if cx > ex: ex = cx
        if cy > ey: ey = cy
    return (sx - margin - margin // 2, sy - margin), (ex + margin // 2, ey + margin)

mouse_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
mouse_idxs = [0, 291, 17, 61]
frame_shape = (740, 580)
bboxs = []
raw_state = False
times = []

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

    face_lms = Face_mesh(image)
    if face_lms:
        (sx, sy), (ex, ey) = face_roi(frame_shape, face_lms, 0)
        bboxs = (sx, sy, ex, ey)

        mouse_lms = [face_lms[n_idx] for n_idx in mouse_idxs]
        abs_mouse_lms = Convert_abs_lms(frame_shape, mouse_lms)
        (m_sx, m_sy), (m_ex, m_ey) = mouse_roi(frame_shape, abs_mouse_lms, 20)
        # print(x1, y1, x2, y2)

    if bboxs:
        face = image[m_sy:m_ey, m_sx:m_ex].copy()
        # face = image[sy:ey, sx:ex].copy()
        face_h, face_w = face.shape[:2]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_blur = cv2.GaussianBlur(face_gray, (5, 5), 1)
        # canny = cv2.Canny(face_gray, 4900, 2000, apertureSize=5, L2gradient=True)
        # canny = cv2.Canny(face_blur, 1500, 1000, apertureSize=5, L2gradient=True)
        canny = cv2.Canny(face_blur, 200, 500, apertureSize=5, L2gradient=True)
        cv2.imshow('canny', canny)
        lines = cv2.HoughLinesP(canny, 1, np.pi/360, 18, minLineLength=20, maxLineGap=0)

        # draw
        if lines is not None:
            raw_state = True
            times.append(time.perf_counter())
            for line in lines:
                for x1, y1, x2, y2 in line:
                    x1 = int(x1 + m_sx)
                    x2 = int(x2 + m_sx)
                    y1 = int(y1 + m_sy)
                    y2 = int(y2 + m_sy)
                    # x1 = int(x1 + sx)
                    # x2 = int(x2 + sx)
                    # y1 = int(y1 + sy)
                    # y2 = int(y2 + sy)
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)

        cv2.rectangle(image, (sx, sy), (ex, ey), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.rectangle(image, (m_sx, m_sy), (m_ex, m_ey), (0, 0, 255), 2, cv2.LINE_AA)
        # Draw_landmark(image, abs_mouse_lms)

    cv2.putText(image, f'bow : {raw_state}', (10, frame_shape[1]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    if len(times) >= 2:
        sec = times[-1] - times[0]
        cv2.putText(image, f'{sec:.5f}sec', (frame_shape[0]-200, frame_shape[1]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)

    cv2.imshow('Bow detection', image)
    out.write(image)
    if cv2.waitKey(5) & 0xFF == 27:
        break
    

out.release()
cap.release()
cv2.destroyAllWindows()