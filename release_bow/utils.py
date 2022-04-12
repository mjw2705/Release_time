import numpy as np
import cv2
import mediapipe as mp


mouse_idx = [0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 185, 40, 39, 37]
mouse_idxs = [0, 291, 17, 61]

def facemesh_roi(frame_shape, face_lms, margin):
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

def facedetect_roi(bboxs, frame_shape):
    sx, sy = int(bboxs[0] * frame_shape[0]), int(bboxs[1] * frame_shape[1])
    ex, ey = int((bboxs[0] + bboxs[2]) * frame_shape[0]), int((bboxs[1] + bboxs[3]) * frame_shape[1])

    return sx, sy, ex, ey

def Face_detection(image):
    face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                            min_detection_confidence=0.7)

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)


    if results.detections:
        for detection in results.detections:
            coords = detection.location_data.relative_bounding_box
            bboxs = [coords.xmin, coords.ymin, coords.width, coords.height]
        return bboxs
    return None

def Convert_abs_lms(frame_shape, lms):
    abs_lms = np.array(lms)
    abs_lms[:, 0] = abs_lms[:, 0] * frame_shape[0]
    abs_lms[:, 1] = abs_lms[:, 1] * frame_shape[1]
    return abs_lms[:, :2]

def Draw_landmark(image, lms):
    for lm in lms:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)

def low_pass_filter(cur, prev, detect, gap):
    if detect:
        if abs(prev[0] - cur[0]) < gap:
            cur[0] = prev[0]
        else:
            prev[0] = cur[0]
        if abs(prev[1] - cur[1]) < gap:
            cur[1] = prev[1]
        else:
            prev[1] = cur[1]
        if abs(prev[2] - cur[2]) < gap:
            cur[2] = prev[2]
        else:
            prev[2] = cur[2]
        if abs(prev[3] - cur[3]) < gap:
            cur[3] = prev[3]
        else:
            prev[3] = cur[3]
    else:
        detect = True
        prev = cur
    
    return cur, prev, detect

def find_line(lines, sx, sy):
    for line in lines:
        for x1, y1, x2, y2 in line:
            try:
                slope = (y2 - y1) // (x2 - x1)
                
            except:
                mom = 1
                slope = (y2 - y1) / mom

            if -20 < slope < -2:
                x1 = int(x1 + sx)
                x2 = int(x2 + sx)
                y1 = int(y1 + sy)
                y2 = int(y2 + sy)
                return x1, x2, y1, y2
            return None, None, None, None