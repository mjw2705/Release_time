import cv2
import mediapipe as mp


class FaceMeshDetector:
    def __init__(self, frame_shape):
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1,
                                                    refine_landmarks=True,
                                                    min_detection_confidence=0.6,
                                                    min_tracking_confidence=0.5)
        self.face_lms = None
        self.width = frame_shape[0]
        self.height = frame_shape[1]

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face.process(rgb)
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                self.face_lms = [[lms.x, lms.y, lms.z] for lms in landmarks.landmark] #len 468
            return len(self.face_lms) > 0
        return False

    def get_face_rect(self):
        if self.face_lms is not None:
            sx, sy = self.width, self.height
            ex, ey = 0, 0
            for lms in self.face_lms:
                cx, cy = int(lms[0] * self.width), int(lms[1] * self.height)
                if cx < sx: sx = cx
                if cy < sy: sy = cy
                if cx > ex: ex = cx
                if cy > ey: ey = cy
            return [max(sx, 0), max(sy, 0), min(ex, self.width), min(ey, self.height)]
            
        return [0, 0, 1, 1]
    
    def get_landmarks(self):
        return self.face_lms