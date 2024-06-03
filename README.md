# 한국 수어 지문자 인식 앱 구현을 위한 딥러닝 모델 비교 분석 및 응용
### 19101151 김규태

## 프로젝트 개요
이 프로젝트는 한국 수어(Korean Sign Language, KSL) 지문자 인식을 위해 딥러닝 모델을 비교하고 최적의 모델을 선택하여 실시간 인식 애플리케이션을 개발하는 것을 목표로 합니다. 본 연구에서는 Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU), Bidirectional LSTM (Bi-LSTM), Bidirectional GRU (Bi-GRU), 1차원 Convolutional Neural Networks (1D-CNN) 모델을 사용하였으며, 각 모델의 성능을 비교하였습니다.

## 데이터셋 준비
- **데이터 수집**: Mediapipe Holistic Library와 OpenCV를 사용하여 31개의 한국 수어 지문자에 대한 영상을 촬영.
- **데이터 전처리**: 각 프레임에서 손가락 관절의 x, y, z 좌표를 추출하여 Numpy 형식의 파일로 저장.
- **데이터 분할**: 데이터셋을 80:20 비율로 Training set과 Validation set으로 분할.

## 모델 설명
- **LSTM**: 시계열 데이터의 장기 종속성을 학습하는 데 사용되는 모델.
- **GRU**: LSTM과 유사하지만 계산 비용이 적고, 학습 속도가 빠른 모델.
- **Bi-LSTM**: 양방향으로 시계열 데이터를 처리하여 더 많은 문맥 정보를 학습하는 모델.
- **Bi-GRU**: GRU의 양방향 버전으로, 효율성과 성능을 동시에 제공.
- **1D-CNN**: 시계열 데이터의 공간적 패턴을 학습하는 데 사용되는 합성곱 신경망 모델.

## 모델 학습 및 평가
- **학습 과정**: 각 모델은 Adam 옵티마이저와 categorical cross-entropy 손실 함수를 사용하여 학습.
- **평가 지표**: 정확도(Accuracy)와 F1 스코어를 사용하여 각 모델의 성능을 평가.
- **결과**: 1D-CNN 모델이 가장 높은 정확도와 F1 스코어를 기록.

## 모델 변환 및 배포
- **TensorFlow Lite 변환**: 학습된 모델을 TensorFlow Lite 형식으로 변환하여 경량화.
- **실시간 애플리케이션 구현**: 변환된 모델을 Python 기반의 실시간 수어 인식 응용 프로그램에 통합.

## 파일 설명
- **Create_Video.py**: 수어 데이터 수집을 위한 영상을 녹화하는 코드
- **Create_Dataset.py**: 녹화한 영상으로부터 각 관절 landmark의 프레임 당 x,y,z좌표 numpy 데이터를 추출하는 코드
- **train_and_compare_sign_language.ipynb**: 다양한 딥러닝 모델을 학습시키고 평가하는 주피터노트북 파일.
- **app.py**: 실시간 수어 인식 애플리케이션의 메인 코드.

## 사용 방법
1. **데이터 준비**:
    - Mediapipe와 OpenCV를 설치하고 `data_preparation.py`를 실행하여 데이터를 수집하고 전처리합니다.
2. **모델 학습**:
    - `model_training.py`를 실행하여 각 모델을 학습시키고 성능을 평가합니다.
3. **모델 변환**:
    - `model_conversion.py`를 실행하여 학습된 모델을 TensorFlow Lite 형식으로 변환합니다.
4. **실시간 인식 앱 실행**:
    - `SLC_Dynamic_models`를 실행하여 실시간 수어 인식 애플리케이션을 실행합니다. LSTM, GRU, 1D-CNN, Bi-LSTM, Bi-GRU 각각의 모델을 선택하여 수어 지문자 동작을 식별할 수 있습니다.

## 필요 라이브러리
- TensorFlow
- Mediapipe
- OpenCV
- Numpy
- Matplotlib
- Scikit-learn

## 결과 및 향후 연구
- **결과**: 1D-CNN 모델이 가장 우수한 성능을 보였으며, GRU와 함께 학습 중 높은 안정성을 보였습니다.
- **향후 연구**: 더 많은 수어 동작과 변형을 포함한 데이터셋 확장, 실시간 구현 및 실용적 시나리오에서의 테스트를 통해 포괄적이고 신뢰할 수 있는 수화 인식 시스템을 개발할 것입니다.

## 참고 자료
- [TensorFlow](https://www.tensorflow.org/)
- [Mediapipe](https://google.github.io/mediapipe/)
- [OpenCV](https://opencv.org/)
