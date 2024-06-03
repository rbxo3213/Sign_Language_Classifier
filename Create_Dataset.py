import cv2
import os
import time
import numpy as np
import mediapipe as mp
from modules.utils import create_directory, vector_normalization
import modules.mediapipe_holistic_module as hm
from tqdm import tqdm

# 디렉토리 생성
create_directory('dataset/Npy모음')

# 학습할 액션 리스트 (한글 지문자 + 숫자 0~9)
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']
seq_length = 10
max_length = 14  # 최대 14초로 고정

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# HolisticDetector 설정
detector = hm.HolisticDetector(min_detection_confidence=0.3)

# 데이터셋을 npy 파일로 저장
dataset = dict()
for i in range(len(actions)):
    dataset[i] = []

videoFolderPath = "dataset/output_video"
videoTestList = os.listdir(videoFolderPath)
testTargetList = []

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)

testTargetList = sorted(testTargetList, key=lambda x: x[x.find("/", 9) + 1], reverse=True)

created_time = int(time.time())

# 진행도를 보기 위한 tqdm 사용
for target in tqdm(testTargetList, desc="Processing videos"):
    data = []
    idx = actions.index(target.split('/')[-2])
    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        print(f"Camera open failed for {target}!")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(max_length * fps)
    
    while True:
        ret, img = cap.read()
        if not ret:
            break
        
        img = detector.find_holistic(img, draw=True)
        _, right_hand_lmList = detector.find_hand_landmarks(img, hand='right')

        if right_hand_lmList is not None:
            joint = np.zeros((21, 2))
            for j, lm in enumerate(right_hand_lmList.landmark):
                joint[j] = [lm.x, lm.y]

            vector, angle_label = vector_normalization(joint)
            angle_label = np.append(angle_label, idx)
            d = np.concatenate([vector.flatten(), angle_label.flatten()])
            data.append(d)
        
        if len(data) >= max_frames:
            break

    cap.release()

    if len(data) == 0:
        print(f"No landmarks detected for {target}")
        continue  # 패딩을 추가할 수 없으므로 이 파일은 건너뜀

    if len(data) < max_frames:
        padding = [np.zeros_like(data[0]) for _ in range(max_frames - len(data))]
        data.extend(padding)
    else:
        data = data[:max_frames]

    data = np.array(data)
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])

for i in range(len(actions)):
    save_data = np.array(dataset[i])
    np.save(os.path.join('dataset', 'Npy모음', f'seq_{actions[i]}'), save_data)

print("Dataset creation complete.")
