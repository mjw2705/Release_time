# pyinstaller --icon=icon-archery.ico main_thread.py
import cv2
import sys
import json
import os
from body.body_main import BodyThread
from bow.bow_main import BowThread

from mp_detection.face import FaceDetector


def load_config(path):
    with open(path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config

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

    elif use_camera:
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if use_video == use_camera:
        sys.exit("video and camera can't run concurrently")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)

    success, image = cap.read()
    resize_image = cv2.resize(image, size)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000/fps)

    face = FaceDetector(size)
    
    if detect_bow and not detect_body:
        use_thread = False
        bow_thread = BowThread(size, resize_image, use_thread)

    elif detect_body and not detect_bow:
        use_thread = False
        body_thread = BodyThread(size, resize_image, use_thread)
    
    elif detect_body and detect_bow:
        use_thread = True
        body_thread = BodyThread(size, resize_image, use_thread)
        bow_thread = BowThread(size, resize_image, use_thread)
        # body_thread.start()
        # bow_thread.start()
    else:
        sys.exit("not select bow or body")

    print('Exit to press Esc button')
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, size)

        if not ret:
            print("Ignoring empty camera frame.")
            break
        
        is_face = face.process(frame)
        frame_origin = frame.copy()

        if detect_bow:
            bow_thread.process(frame, is_face, delay)
        if detect_body:
            body_thread.process(frame_origin, is_face, delay)

        if detect_bow and detect_body:
            if body_thread.is_close or bow_thread.is_close:
                break
        
        if cv2.waitKey(delay) == 27:
            break

    if detect_bow:
        bow_thread.close()
    if detect_body:
        body_thread.close()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()