import sys
import threading
from PySide6.QtCore import QSize, Qt, Signal, QObject, QTimer
from PySide6.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, 
                               QHBoxLayout, QWidget, QLabel, QTextEdit, 
                               QFrame, QPushButton, QSpacerItem, QSizePolicy)
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPalette
import cv2
import numpy as np
import tensorflow as tf
from threading import Lock
from PIL import ImageFont, ImageDraw, Image
import time
import os
import mediapipe as mp

os.environ['GEVENT_SUPPORT'] = 'True'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
sys.path.append('modules')
import modules.mediapipe_holistic_module as hm
from modules.utils import vector_normalization

class Communicate(QObject):
    signal = Signal(str, float)
    frame_signal = Signal(np.ndarray)

class SignLanguageRecognitionApp(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Recognition")
        self.setGeometry(100, 100, 1600, 900)  # Increased height to accommodate larger image

        # Signal initialization
        self.result_signal = Communicate()
        self.result_signal.signal.connect(self.update_result)
        self.result_signal.frame_signal.connect(self.update_frame)

        # Initialize UI
        self.init_ui()

        # Initialize recognition thread
        self.recognition_thread = threading.Thread(target=self.recognize_gestures, daemon=True)
        self.recognition_thread.start()

        # Initialize frame update timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)  # Update every 100ms (10 fps)

        self.current_frame = None
        self.typed_text = ""
        self.last_consonant = ""
        self.last_vowel = ""
        self.last_final = ""
        self.last_recognized_time = time.time()
        self.last_output_time = time.time()

        # Add space if no recognition in 3 seconds
        self.space_timer = QTimer()
        self.space_timer.timeout.connect(self.add_space_if_no_recognition)
        self.space_timer.start(1000)  # Check every second

        # Load models
        self.load_models()

        # Initialize variables for recognition modes
        self.sequence = []

    def init_ui(self):
        # Title
        title_label = QLabel("Sign Language Recognition", self)
        title_label.setFont(QFont("Arial", 24, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)

        # Translated text
        self.translated_text = QTextEdit(self)
        self.translated_text.setReadOnly(True)
        self.translated_text.setFont(QFont("Arial", 20))
        self.translated_text.setFrameShape(QFrame.NoFrame)
        self.translated_text.setStyleSheet("background-color: #f3f3f3; border: none;")

        # Video feed
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("border: 2px solid #4CAF50; background-color: white;")

        # Sign language image
        self.sign_language_image_label = QLabel(self)
        self.sign_language_image_label.setAlignment(Qt.AlignCenter)
        self.sign_language_image_label.setStyleSheet("border: 2px solid #4CAF50; background-color: white;")
        self.sign_language_image_label.setFixedSize(600, 600)
        self.sign_language_image_label.setVisible(False)  # Initially hidden

        # Result label
        self.result_label = QLabel("Mode: Fingerspelling", self)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setFont(QFont("Arial", 18))
        self.result_label.setStyleSheet("color: #333; margin-bottom: 20px;")

        # Exit button
        exit_button = QPushButton("Exit", self)
        exit_button.setFixedSize(100, 40)
        exit_button.setFont(QFont("Arial", 14))
        exit_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px;")
        exit_button.clicked.connect(self.close)

        # Show SL image button
        show_image_button = QPushButton("SL Image", self)
        show_image_button.setFixedSize(150, 40)
        show_image_button.setFont(QFont("Arial", 14))
        show_image_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px;")
        show_image_button.clicked.connect(self.toggle_sl_image)

        # Reset text button
        reset_text_button = QPushButton("Reset", self)
        reset_text_button.setFixedSize(100, 40)
        reset_text_button.setFont(QFont("Arial", 14))
        reset_text_button.setStyleSheet("background-color: #4CAF50; color: white; border: none; border-radius: 5px;")
        reset_text_button.clicked.connect(self.reset_text)

        # Layout settings
        layout = QVBoxLayout()
        layout.addWidget(title_label)

        video_layout = QHBoxLayout()
        video_layout.addWidget(self.video_label, alignment=Qt.AlignCenter)
        video_layout.addWidget(self.sign_language_image_label, alignment=Qt.AlignCenter)

        layout.addLayout(video_layout)
        layout.addWidget(self.result_label)
        layout.addWidget(self.translated_text)

        layout.addWidget(show_image_button, alignment=Qt.AlignCenter)
        layout.addWidget(reset_text_button, alignment=Qt.AlignCenter)
        layout.addWidget(exit_button, alignment=Qt.AlignCenter)

        layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # Set the overall style
        self.setStyleSheet("""
        QMainWindow {
            background-color: #f3f3f3;
        }
        QLabel {
            color: #333;
        }
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px;
        }
        """)

        # Apply dark mode palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#121212"))
        palette.setColor(QPalette.WindowText, QColor("#FAFAFA"))
        palette.setColor(QPalette.Base, QColor("#1E1E1E"))
        palette.setColor(QPalette.Text, QColor("#FAFAFA"))
        palette.setColor(QPalette.Button, QColor("#4CAF50"))
        palette.setColor(QPalette.ButtonText, QColor("#FAFAFA"))
        self.setPalette(palette)

    def load_models(self):
        # Load fingerspelling model
        self.fingerspelling_interpreter = tf.lite.Interpreter(model_path="models/sign_language_classifier_cnn.tflite")
        self.fingerspelling_interpreter.allocate_tensors()
        self.fingerspelling_input_details = self.fingerspelling_interpreter.get_input_details()
        self.fingerspelling_output_details = self.fingerspelling_interpreter.get_output_details()

        # Initialize MediaPipe holistic model
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

    def toggle_sl_image(self):
        if self.sign_language_image_label.isVisible():
            self.sign_language_image_label.setVisible(False)
        else:
            pixmap = QPixmap("signlangue_image.jpg").scaled(600, 600, Qt.KeepAspectRatio)
            self.sign_language_image_label.setPixmap(pixmap)
            self.sign_language_image_label.setVisible(True)

    def reset_text(self):
        self.typed_text = ""
        self.translated_text.setPlainText(self.typed_text)
        self.sequence = []

    def update_result(self, action, confidence):
        self.result_label.setText(f"Result: {action} (Confidence: {confidence:.2f})")
        current_time = time.time()
        if action != "?" and (current_time - self.last_output_time > 2):  # 2 seconds delay for the same character
            self.add_korean_letter(action)
            self.last_output_time = current_time
        self.last_recognized_time = current_time

    def add_space_if_no_recognition(self):
        if time.time() - self.last_recognized_time > 3:
            self.typed_text += " "
            self.translated_text.setPlainText(self.typed_text)
            self.last_recognized_time = time.time()

    def add_korean_letter(self, action):
        # 자음과 모음 결합 로직
        consonants = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
        vowels = 'ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅒㅔㅖㅚㅟㅢ'
        finals = 'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'

        if action in consonants:
            if self.last_vowel:
                # 받침이 이미 있는 경우
                if self.last_final:
                    self.typed_text += action
                    self.last_consonant = action
                    self.last_vowel = ""
                    self.last_final = ""
                else:
                    self.last_final = action
                    combined = self.combine_hangul(self.last_consonant, self.last_vowel, self.last_final)
                    self.typed_text = self.typed_text[:-1] + combined
                    self.last_consonant = ""
                    self.last_vowel = ""
                    self.last_final = ""
            else:
                self.last_consonant = action
                self.typed_text += action
        elif action in vowels:
            if self.last_vowel:
                # 받침이 있는 글자에 모음이 오면 새로운 글자
                if self.last_final:
                    if self.last_vowel == "ㅗ" and action == "ㅣ":  # ㅚ를 위한 예외처리
                        combined = self.combine_hangul(self.last_consonant, "ㅚ", "")
                        self.typed_text = self.typed_text[:-1] + combined
                        self.last_consonant = ""
                        self.last_vowel = ""
                        self.last_final = ""
                    elif self.last_vowel == "ㅜ" and action == "ㅣ":  # ㅟ를 위한 예외처리
                        combined = self.combine_hangul(self.last_consonant, "ㅟ", "")
                        self.typed_text = self.typed_text[:-1] + combined
                        self.last_consonant = ""
                        self.last_vowel = ""
                        self.last_final = ""
                    elif self.last_vowel == "ㅡ" and action == "ㅣ":  # ㅢ를 위한 예외처리
                        combined = self.combine_hangul(self.last_consonant, "ㅢ", "")
                        self.typed_text = self.typed_text[:-1] + combined
                        self.last_consonant = ""
                        self.last_vowel = ""
                        self.last_final = ""
                    else:
                        self.typed_text = self.typed_text[:-1] + self.last_consonant + action
                        self.last_consonant = self.last_final
                        self.last_vowel = action
                        self.last_final = ""
                else:
                    combined = self.combine_hangul(self.last_consonant, action)
                    self.typed_text = self.typed_text[:-1] + combined
                    self.last_vowel = action
                    self.last_consonant = ""
            else:
                if self.last_consonant:
                    combined = self.combine_hangul(self.last_consonant, action)
                    self.typed_text = self.typed_text[:-1] + combined
                else:
                    self.typed_text += action
                self.last_vowel = action
        else:
            self.typed_text += action
            self.last_consonant = ""
            self.last_vowel = ""
            self.last_final = ""
        self.translated_text.setPlainText(self.typed_text)

    def combine_hangul(self, consonant, vowel, final=""):
        consonant_list = 'ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ'
        vowel_list = 'ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ'
        final_list = 'ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ'

        consonant_index = consonant_list.index(consonant)
        vowel_index = vowel_list.index(vowel)
        final_index = final_list.index(final) + 1 if final else 0

        return chr(0xAC00 + consonant_index * 588 + vowel_index * 28 + final_index)

    def update_frame(self, frame):
        self.current_frame = frame

    def update_ui(self):
        if self.current_frame is not None:
            img_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            bytes_per_line = ch * w
            q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            q_pixmap = QPixmap(q_image)
            self.video_label.setPixmap(q_pixmap)

    def recognize_gestures(self):
        fontpath = "static/fonts/HMKMMAG.TTF"
        font = ImageFont.truetype(fontpath, 40)

        fingerspelling_seq_length = 10
        fingerspelling_actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
                                  'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
                                  'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

        detector = hm.HolisticDetector(min_detection_confidence=0.3)
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            start_time = time.time()
            ret, img = cap.read()
            if not ret:
                break

            img = detector.find_holistic(img, draw=True)
            _, right_hand_lmList = detector.find_hand_landmarks(img)

            if right_hand_lmList is not None:
                joint = np.zeros((21, 2))
                for j, lm in enumerate(right_hand_lmList.landmark):
                    joint[j] = [lm.x, lm.y]

                vector, angle_label = vector_normalization(joint)
                d = np.concatenate([vector.flatten(), angle_label.flatten()])
                self.sequence.append(d)

                if len(self.sequence) >= fingerspelling_seq_length:
                    self.sequence = self.sequence[-fingerspelling_seq_length:]

                    input_data = np.expand_dims(np.array(self.sequence, dtype=np.float32), axis=0)
                    self.fingerspelling_interpreter.set_tensor(self.fingerspelling_input_details[0]['index'], input_data)
                    self.fingerspelling_interpreter.invoke()

                    y_pred = self.fingerspelling_interpreter.get_tensor(self.fingerspelling_output_details[0]['index'])
                    i_pred = int(np.argmax(y_pred[0]))
                    conf = y_pred[0][i_pred]

                    if conf < 0.9:
                        self.result_signal.frame_signal.emit(img)
                        continue

                    action = fingerspelling_actions[i_pred]
                    self.result_signal.signal.emit(action, conf)

            self.result_signal.frame_signal.emit(img)

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Frame Processing Time: {elapsed_time:.3f} seconds")

        cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SignLanguageRecognitionApp()
    window.show()
    sys.exit(app.exec())
