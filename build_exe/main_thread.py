# pyinstaller --icon=icon-archery.ico main_thread.py
import cv2
import sys
import json
import os
import mediapipe as mp
from body.body_main import BodyThread
from bow.bow_main import BowThread


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

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


def main():
    rootPath = os.path.dirname(os.path.abspath(__file__))
    FPS = 30

    config = load_config('config.json')
    use_video = config["use_video"]
    video_path = config["video_path"]
    use_camera = config["use_camera"]
    camera_id = config["camera_id"]
    detect_bow = config["detect_bow"]
    detect_body = config["detect_body"]
    display_size = config["display_size"]

    width, height = display_size
    size = (width, height)

    if use_video:
        abs_videoPath = rootPath + '/' + video_path
        cap = cv2.VideoCapture(abs_videoPath)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = round(1000/fps)

        success, image = cap.read()
        resize_image = cv2.resize(image, size)
        bbox = Face_detection(image)

    elif use_camera:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 30)

        success, image = cap.read()
        resize_image = cv2.resize(image, size)
        bbox = Face_detection(image)

    if use_video == use_camera:
        sys.exit("video and camera can't run concurrently")

    if detect_bow and not detect_body:
        use_thread = False
        bow_thread = BowThread(size, resize_image, bbox, use_thread)

    elif detect_body and not detect_bow:
        use_thread = False
        body_thread = BodyThread(size, resize_image, bbox, use_thread)
    
    elif detect_body and detect_bow:
        use_thread = True
        body_thread = BodyThread(size, resize_image, bbox, use_thread)
        bow_thread = BowThread(size, resize_image, bbox, use_thread)

    else:
        sys.exit("not select bow or body")

    print('Exit to press Esc button')

    num = 1 if use_camera else delay
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, size)

        if not ret:
            print("Ignoring empty camera frame.")
            break

        bbox = Face_detection(frame)
        frame_origin = frame.copy()

        if detect_bow:
            bow_thread.process(frame, bbox, num)
        if detect_body:
            body_thread.process(frame_origin, bbox, num)

        if detect_bow and detect_body:
            if body_thread.is_close or bow_thread.is_close:
                break
        
        if cv2.waitKey(num) == 27:
            break

    if detect_bow:
        bow_thread.close()
    if detect_body:
        body_thread.close()
            
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()