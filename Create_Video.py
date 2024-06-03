import cv2
import sys, os
import time
import mediapipe as mp
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from modules.utils import create_directory, vector_normalization
import modules.mediapipe_holistic_module as hm
import json

# 폰트 설정
fontpath = "fonts/HMKMMAG.TTF"
font = ImageFont.truetype(fontpath, 40)

# 디렉토리 생성
create_directory('dataset')
create_directory('dataset/output_video')

# 학습할 액션 리스트 (한글 지문자 + 숫자 0~9)
actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
           'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
           'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ',
           '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
secs_for_action = 30
seq_length = 10

# MediaPipe hands model
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.3, min_tracking_confidence=0.5)

# HolisticDetector 설정
detector = hm.HolisticDetector(min_detection_confidence=0.3)

cap = cv2.VideoCapture(0)
created_time = int(time.time())

# 열렸는지 확인
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()

# 웹캠의 속성 값을 받아오기
w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
delay = round(1000/fps) if fps != 0 else round(1000/30)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

# 프레임을 받아와서 저장하기
while cap.isOpened():
    for idx, action in enumerate(actions):
        os.makedirs(f'dataset/output_video/{action}', exist_ok=True)
        videoFolderPath = f'dataset/output_video/{action}'
        videoList = sorted(os.listdir(videoFolderPath), key=lambda x:int(x[x.find("_")+1:x.find(".")]))
      
        take = 1 if len(videoList) == 0 else int(videoList[-1].split('_')[1].split('.')[0]) + 1
        saved_video_path = f'dataset/output_video/{action}/{action}_{take}.avi'
        out = cv2.VideoWriter(saved_video_path, fourcc, fps, (w, h))

        ret, img = cap.read()
        if not ret:
            break
         
        # 한글 폰트 출력    
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((10, 30), f'Ready for input {action.upper()}', font=font, fill=(255, 255, 255))
        img = np.array(img_pil)
        cv2.imshow('img', img)
        cv2.waitKey(4000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()
            if not ret:
                break
            
            # 비디오 녹화
            out.write(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = hands.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if result.multi_hand_landmarks is not None:
                for res in result.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('img', img)

            # esc를 누르면 강제 종료
            if cv2.waitKey(delay) == 27: 
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)

for target in testTargetList:
    data = []
    idx = actions.index(target.split('/')[-2])
    cap = cv2.VideoCapture(target)

    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = round(1000/fps) if fps != 0 else round(1000/30)

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

        cv2.waitKey(delay)
        if cv2.waitKey(delay) == 27: 
            break

    data = np.array(data)
    for seq in range(len(data) - seq_length):
        dataset[idx].append(data[seq:seq + seq_length])

for i in range(len(actions)):
    save_data = np.array(dataset[i])
    np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)

print("Dataset creation complete.")
