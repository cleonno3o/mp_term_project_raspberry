import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, Response
import threading

app = Flask(__name__)

cap = None
lock = threading.Lock()  # 동기화 Lock
selected_camera = 0  # 기본 카메라 인덱스

# 학습된 모델 로드
model_path = "digit_recognition_model.h5"  # 모델 경로
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# 전처리 함수 정의
def preprocess_image(image):
    """
    이미지를 모델 입력 형식에 맞게 전처리
    - 흑백 변환
    - 크기 조정
    - 정규화
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    image = cv2.resize(image, (28, 28))  # 모델 입력 크기로 조정
    image = image.astype("float32") / 255.0  # 정규화
    image = image.reshape(1, 28, 28, 1)  # 배치 차원 추가
    return image


def list_available_cameras():
    """사용 가능한 카메라 인덱스를 확인"""
    index = 0
    available_cameras = []
    while True:
        temp_cap = cv2.VideoCapture(index)
        if temp_cap.isOpened():
            available_cameras.append(index)
            temp_cap.release()
        else:
            break
        index += 1
    return available_cameras


def set_camera(index):
    """카메라를 선택하고 초기화"""
    global cap
    with lock:
        if cap is not None:
            cap.release()  # 기존 카메라 해제
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            print(f"[ERROR] Camera index {index} could not be opened.")
            cap = None
        else:
            print(f"[INFO] Camera index {index} successfully opened.")


def generate_frames():
    """번호 인식 결과를 포함한 프레임 생성"""
    global cap
    while True:
        with lock:
            if cap is None or not cap.isOpened():
                print("[ERROR] No camera available.")
                break
            success, frame = cap.read()
        if not success:
            print("[ERROR] Failed to read frame.")
            break
        else:
            # 화면 중앙에 ROI 설정
            height, width, _ = frame.shape
            roi_size = 200  # ROI 크기
            x_center, y_center = width // 2, height // 2
            x_start = x_center - roi_size // 2
            y_start = y_center - roi_size // 2
            roi = frame[y_start:y_start + roi_size, x_start:x_start + roi_size]

            # ROI에 박스 그리기
            cv2.rectangle(frame, (x_start, y_start), (x_start + roi_size, y_start + roi_size), (0, 255, 0), 2)

            # 전처리 및 모델 예측
            processed_image = preprocess_image(roi)
            prediction = model.predict(processed_image)
            predicted_digit = np.argmax(prediction)
            confidence = prediction[0][predicted_digit] * 100

            # 예측 결과 화면에 표시
            label = f"Digit: {predicted_digit}, Confidence: {confidence:.2f}%"
            cv2.putText(frame, label, (x_start, y_start - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            # 프레임 인코딩
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video_feed():
    """비디오 스트림"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def terminal_select_camera():
    """터미널에서 카메라 선택"""
    available_cameras = list_available_cameras()
    if not available_cameras:
        print("[ERROR] No cameras found.")
        return None

    print("Available cameras:")
    for idx in available_cameras:
        print(f" - Camera {idx}")

    while True:
        try:
            selected = int(input("Enter the camera index to use: "))
            if selected in available_cameras:
                return selected
            else:
                print("[ERROR] Invalid selection. Try again.")
        except ValueError:
            print("[ERROR] Please enter a valid number.")


if __name__ == '__main__':
    try:
        # 터미널에서 카메라 선택
        selected_camera = terminal_select_camera()
        if selected_camera is not None:
            set_camera(selected_camera)
            print(f"[INFO] Camera {selected_camera} selected.")
            app.run(host='0.0.0.0', port=33389)
        else:
            print("[ERROR] No camera selected. Exiting.")
    finally:
        if cap:
            cap.release()
