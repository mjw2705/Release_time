import cv2
import mediapipe as mp


class FaceDetector:
    def __init__(self, frame_shape):
        self.face = mp.solutions.face_detection.FaceDetection(model_selection=0,
                                                            min_detection_confidence=0.5)
        self.bbox = None
        self.width = frame_shape[0]
        self.height = frame_shape[1]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face.process(rgb)
        if results.detections:
            for detection in results.detections:
                coords = detection.location_data.relative_bounding_box
                self.bbox = [coords.xmin, coords.ymin, coords.width, coords.height]
            return len(self.bbox) > 0
        return False

    def get_face_rect(self):
        if self.bbox is not None:
            sx, sy = int(self.bbox[0] * self.width), int(self.bbox[1] * self.height)
            ex, ey = int((self.bbox[0] + self.bbox[2]) * self.width), int((self.bbox[1] + self.bbox[3]) * self.height)

            return [max(sx, 0), max(sy, 0), min(ex, self.width), min(ey, self.height)]
        return [0, 0, 1, 1]