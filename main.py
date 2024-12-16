import time
import numpy as np
import tensorflow as tf
from gpiozero import DistanceSensor
import cv2
from collections import Counter
import serial

# 초음파 센서 핀 설정
ECHO_PIN = 23
TRIGGER_PIN = 24

# 숫자 인식 모델 경로
MODEL_PATH = "digit_recognition_model.h5"

# ROI 크기 설정
ROI_SIZE = 200

# 거리 임계값
DISTANCE_THRESHOLD = 50

# 숫자 인식 반복 횟수
RECOGNITION_COUNT = 10

# UART 설정
UART_PORT = "/dev/serial0"
UART_BAUDRATE = 9600
UART_TIMEOUT = 1

# 초음파 센서를 통한 거리 측정
def check_distance(threshold=DISTANCE_THRESHOLD):
    sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN)
    distance_cm = sensor.distance * 100  # 거리 계산 (cm)
    print(f"Distance: {distance_cm:.2f} cm")
    return distance_cm <= threshold

# 이미지 전처리 함수
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype('float32') / 255.0
    gray = gray.reshape(1, 28, 28, 1)
    return gray

# 숫자 인식 함수
def recognize_digit(frame, model):
    # 화면 중심에 ROI 설정
    height, width = frame.shape[:2]
    x_center, y_center = width // 2, height // 2
    x_start = x_center - ROI_SIZE // 2
    y_start = y_center - ROI_SIZE // 2
    roi = frame[y_start:y_start + ROI_SIZE, x_start:x_start + ROI_SIZE]
    cv2.imshow("ROI", roi)

    # 전처리 후 예측
    preprocessed_roi = preprocess_image(roi)
    predictions = model.predict(preprocessed_roi)
    predicted_label = np.argmax(predictions)
    return predicted_label

# UART 송신 함수
def send_to_uart(uart, data):
    uart.write(data.encode('utf-8'))
    print(f"Sent to UART: {data}")

# UART 수신 함수
def receive_from_uart(uart):
    data = uart.read_until().decode('utf-8').strip()
    print(f"Received from UART: {data}")
    return data

# 메인 동작 함수
def main():
    # 숫자 인식 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH)

    # UART 초기화
    uart = serial.Serial(port=UART_PORT, baudrate=UART_BAUDRATE, timeout=UART_TIMEOUT)

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    while True:
        print("Waiting for object to enter the detection range...")

        # 초음파 센서로 거리 확인
        while not check_distance():
            time.sleep(0.1)  # 센서 체크 주기

        print("Object detected! Starting digit recognition.")

        # 숫자 인식 결과 저장
        recognized_digits = []

        for _ in range(RECOGNITION_COUNT):
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            # 숫자 인식 및 결과 저장
            recognized_digit = recognize_digit(frame, model)
            recognized_digits.append(recognized_digit)
            print(f"Recognized digit: {recognized_digit}")
            cv2.waitKey(500)  # 0.5초 간격

        # 최빈값 계산
        if recognized_digits:
            most_frequent_digit = Counter(recognized_digits).most_common(1)[0][0]
            print(f"Most frequently recognized digit: {most_frequent_digit}")

            # UART로 전송
            send_to_uart(uart, str(most_frequent_digit))

            # UART 수신 대기
            while True:
                #S32K144 에서 모듈 동작후 동작완료 신호 대기 
                response = receive_from_uart(uart)
                #동작완료신호 "DONE"을 수신받으면 초기로 돌아감.
                if response == "DONE":
                    print("Operation completed. Restarting detection...")
                    break
        else:
            print("No digits recognized. Restarting detection...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
