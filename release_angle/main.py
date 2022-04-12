# pyinstaller --icon=icon-archery.ico release_angle.py
import time
import cv2
import numpy as np
import mediapipe as mp

from utils import *



mp_holistic = mp.solutions.holistic
holistic = mp.solutions.holistic.Holistic(static_image_mode=False,
                                        min_detection_confidence=0.4,
                                        min_tracking_confidence=0.4)

pose = mp.solutions.pose.Pose(min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)


frame_shape = (640, 480)
frame_w, frame_h = frame_shape
half_w = frame_w // 2

backboard = np.zeros((frame_h * 2, frame_w, 3), np.uint8)

before_idx = 0
before_angle = 0
left_angle_list = [0] * (frame_w // 2)
right_angle_list = [0] * (frame_w // 2)

angle_record = []
release_state = False
left_angle, right_angle = 0, 0
prev_left_angle, prev_right_angle = 0, 0

ptime = 0
sec = 0
prev_lms = []
detect = False
times = []

# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap = cv2.VideoCapture('Ahn.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)
cap.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('body.avi', fourcc, 30, (frame_w, frame_h * 2))

print('종료 esc 버튼')

while cap.isOpened():
    leftboard = np.zeros((frame_h, half_w, 3), np.uint8)
    rightboard = np.zeros((frame_h, half_w, 3), np.uint8)
    rightboard[:, 0] = [0, 0, 255]
    ret, image = cap.read()
    # image = cv2.flip(image, 1)
    image = cv2.resize(image, (frame_w, frame_h))
    if not ret:
        print("Ignoring empty camera frame.")
        continue

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    pose_lms, pose_abs_lms = poses(results, frame_shape)
    # results = holistic.process(rgb)
    # pose_lms, pose_abs_lms = holistics(results, frame_shape)

    if pose_abs_lms is not None:
        pose_abs_lms, prev_lms, detect = low_pass_filter(pose_abs_lms, prev_lms, detect, 10)
        # 각도 계산
        left_angle = angle_calc(pose_abs_lms, left_idx)
        right_angle = angle_calc(pose_abs_lms, right_idx)
        left_angle_list.append(left_angle)
        right_angle_list.append(right_angle)
        
    else:
        left_angle, right_angle = 0, 0
        left_angle_list.append(0)
        right_angle_list.append(0)
    
    if prev_left_angle == left_angle and prev_right_angle == right_angle:
        angle_record.append([left_angle, right_angle])
        times.append(time.perf_counter())
    else:
        prev_left_angle = left_angle
        prev_right_angle = right_angle
        angle_record.clear()
        times.clear()

    left_angle_list, _, _ = get_angle_board(leftboard, frame_shape, left_angle_list, before_angle, before_idx)
    right_angle_list, _, _ = get_angle_board(rightboard, frame_shape, right_angle_list, before_angle, before_idx)

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    if len(angle_record) > 10:
        if len(times) >= 2:
            sec = times[-1] - times[0]

    cv2.putText(image, f'{sec:.5f}sec', (frame_w-200, frame_h-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)

    cv2.putText(image, f'fps:{int(fps)}', (10, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (255, 0, 0), 1)
    putText_graph(leftboard, frame_shape, 'left', left_angle)
    putText_graph(rightboard, frame_shape, 'right', right_angle)

    if pose_lms is not None:
        draw_landmark(image, pose_abs_lms)
        draw_armline(image, pose_abs_lms, left_idx)
        draw_armline(image, pose_abs_lms, right_idx)

    backboard[:frame_h, :frame_w] = image
    backboard[frame_h:, :half_w] = leftboard
    backboard[frame_h:, half_w:] = rightboard

    cv2.imshow('backboard', backboard)
    out.write(backboard)

    if cv2.waitKey(1) == 27:
        break

out.release()
cap.release()
cv2.destroyAllWindows()