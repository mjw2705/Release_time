import cv2
import numpy as np
import mediapipe as mp

frame_shape = (640, 480)
half_width = frame_shape[0] // 2

def holistics(result, frame_shape):
    if result.face_landmarks is None:
        print('No face')
    else:
        pose_lms = np.zeros((33, 4))
        pose_abs_lms = np.zeros((33, 2))

        for j, lm in enumerate(result.pose_landmarks.landmark):
            pose_lms[j] = [lm.x, lm.y, lm.z, lm.visibility]
            pose_abs_lms[j] = [int(lm.x * frame_shape[0]), int(lm.y * frame_shape[1])]
        
        return pose_lms, pose_abs_lms
    return None, None


def poses(results, frame_shape):
    if results.pose_landmarks is None:
        print('No pose')
    else:
        pose_lms = np.zeros((33, 3))
        pose_abs_lms = np.zeros((33, 2))

        for j, lm in enumerate(results.pose_landmarks.landmark):
            pose_lms[j] = [lm.x, lm.y, lm.z]
            pose_abs_lms[j] = [int(lm.x * frame_shape[0]), int(lm.y * frame_shape[1])]

        return pose_lms, pose_abs_lms

    return None, None

def low_pass_filter(cur, prev, detect, gap):
    if detect:
        idx = 0
        for land, prev_land in zip(cur, prev):
            if abs(land[0] - prev_land[0]) < gap:
                cur[idx][0] = prev_land[0]
            else:
                prev[idx][0] = land[0]

            if abs(land[1] - prev_land[1]) < gap:
                cur[idx][1] = prev_land[1]
            else:
                prev[idx][1] = land[1]
            idx += 1
    else:
        detect = True
        prev = cur
    
    return cur, prev, detect

def angle_calc(pose_lms, idxs):
    v1 = pose_lms[[idxs[1], idxs[1]], :3]
    v2 = pose_lms[[idxs[0], idxs[2]], :3]
    v = v1 - v2
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    dot_product = np.dot(v[0], v[1])

    angle = np.arccos(dot_product)
    angle = round(np.degrees(angle))
    
    return angle

def draw_landmark(image, pose_lms):
    for lm in pose_lms[11:23]:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 2, (255, 255, 0), -1, cv2.LINE_AA)

def draw_armline(image, pose_lms, idx):
    cv2.line(
        image, 
        (int(pose_lms[idx[0]][0]), int(pose_lms[idx[0]][1])), 
        (int(pose_lms[idx[1]][0]), int(pose_lms[idx[1]][1])), 
        (255, 0, 0), 
        1, cv2.LINE_AA
    )
    cv2.line(
        image, 
        (int(pose_lms[idx[1]][0]), int(pose_lms[idx[1]][1])), 
        (int(pose_lms[idx[2]][0]), int(pose_lms[idx[2]][1])), 
        (255, 0, 0), 
        1, cv2.LINE_AA
    )

def get_angle_board(graphboard, angle_list, before_angle, before_idx):
    angle_list = angle_list[1:]
    point = np.array([frame_shape[1]-100] * half_width)

    point_angle = point - np.array(angle_list)

    for i, angle_val in enumerate(point_angle):
        graphboard[angle_val][i] = [0, 0, 255]

        if i > 0:
            if before_angle > angle_val:
                graphboard[angle_val:before_angle, i] = [0, 0, 255]
                graphboard[before_angle, before_idx] = [0, 0, 255]
            else:
                graphboard[before_angle:angle_val, i] = [0, 0, 255]
                graphboard[before_angle, before_idx] = [0, 0, 255]

        before_idx = i
        before_angle = angle_val

    return angle_list, before_idx, before_angle

def putText_graph(graphboard, side, angle):
    cv2.putText(graphboard, f'{side}', (10, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
    cv2.putText(graphboard, f'angle:{angle}', (half_width // 2, frame_shape[1]-20), cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0, 0, 255), 1)
