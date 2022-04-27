import numpy as np
import cv2


def get_facebox(frame_shape, faces):
    x1, y1, x2, y2 = faces
    sx = max(0, x1)
    sy = max(0, y1)
    ex = min(frame_shape[0], x2)
    ey = min(frame_shape[1], y2)
    return [sx, sy, ex, ey]

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
    for line in lines[:1]:
        # print(line)
        for x1, y1, x2, y2 in line:
            if (x2 - x1) != 0:
                slope = (y2 - y1) // (x2 - x1)
            else:
                slope = (y2 - y1) // 1

            if -20 < slope < 0 or 0 < slope < 20:
                x1 = int(x1 + sx)
                x2 = int(x2 + sx)
                y1 = int(y1 + sy)
                y2 = int(y2 + sy)
                return [x1, x2, y1, y2]
            return None

def is_not_move(line, prev_line, gap):
    x1, x2, y1, y2 = line
    xline, yline, xline2, yline2 = prev_line

    if abs(xline - x1) < gap or abs(yline - y1) < gap and\
        abs(xline2 - x2) < gap or abs(yline2 - y2) < gap:
        return True
    else:
        return False


#====================================================================================
def Convert_abs_lms(frame_shape, lms):
    abs_lms = np.array(lms)
    abs_lms[:, 0] = abs_lms[:, 0] * frame_shape[0]
    abs_lms[:, 1] = abs_lms[:, 1] * frame_shape[1]
    return abs_lms[:, :2]

def Draw_landmark(image, lms):
    for lm in lms:
        cv2.circle(image, (int(lm[0]), int(lm[1])), 1, (0, 0, 255), -1, cv2.LINE_AA)