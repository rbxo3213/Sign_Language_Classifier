import cv2
import mediapipe as mp
import math

class HolisticDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, smooth_landmarks=True, enable_segmentation=False, smooth_segmentation=True, refine_face_landmarks=False, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            smooth_landmarks=smooth_landmarks,
            enable_segmentation=enable_segmentation,
            smooth_segmentation=smooth_segmentation,
            refine_face_landmarks=refine_face_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_holistic(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.holistic.process(img_rgb)
        if draw and self.results.pose_landmarks:
            self.mp_drawing.draw_landmarks(img, self.results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
            self.mp_drawing.draw_landmarks(img, self.results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        return img

    def find_landmarks(self, img, landmark_type='pose', draw=True):
        landmark_list = []
        landmark_attr = getattr(self.results, f'{landmark_type}_landmarks', None)
        if landmark_attr:
            for lm in landmark_attr.landmark:
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * (w + h) / 2)
                landmark_list.append([cx, cy, cz])
        return landmark_list

    def find_hand_landmarks(self, img, hand='right', draw=True):
        hand_landmarks = getattr(self.results, f'{hand}_hand_landmarks', None)
        landmark_list = []
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                h, w, _ = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * (w + h) / 2)
                landmark_list.append([cx, cy, cz])
        return landmark_list, hand_landmarks

    def fingers_up(self, hand_landmarks, axis=False):
        fingers = []
        if axis:
            if hand_landmarks[self.tip_ids[0]][2] < hand_landmarks[self.tip_ids[0] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
            for id in range(1, 5):
                if hand_landmarks[self.tip_ids[0]][1] < hand_landmarks[self.tip_ids[4]][1]:
                    fingers.append(1 if hand_landmarks[self.tip_ids[id]][1] > hand_landmarks[self.tip_ids[id] - 2][1] else 0)
                else:
                    fingers.append(1 if hand_landmarks[self.tip_ids[id]][1] < hand_landmarks[self.tip_ids[id] - 2][1] else 0)
        else:
            for id in range(5):
                if id == 0:
                    fingers.append(1 if hand_landmarks[self.tip_ids[id]][1] < hand_landmarks[self.tip_ids[id] - 2][1] else 0)
                else:
                    fingers.append(1 if hand_landmarks[self.tip_ids[id]][2] < hand_landmarks[self.tip_ids[id] - 2][2] else 0)
        return fingers

    def find_center(self, p1, p2, landmarks):
        x1, y1 = landmarks[p1][:2]
        x2, y2 = landmarks[p2][:2]
        return (x1 + x2) // 2, (y1 + y2) // 2

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3, landmarks=None):
        x1, y1 = landmarks[p1][:2]
        x2, y2 = p2 if isinstance(p2, (list, tuple)) else landmarks[p2][:2]
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
        return math.hypot(x2 - x1, y2 - y1), img

    def find_angle(self, img, p1, p2, p3, landmarks, draw=True):
        x1, y1 = landmarks[p1][:2]
        x2, y2 = landmarks[p2][:2]
        x3, y3 = landmarks[p3][:2]
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
        angle += 360 if angle < 0 else 0
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle
