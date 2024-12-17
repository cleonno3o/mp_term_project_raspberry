# import serial
# def read_uart():

#     ser = serial.Serial(
#         port='/dev/ttyAMA2', 
#         baudrate=9600,   
#         timeout=1  
#     )

#     try:
#         print("Listening on UART (AMA2)...")
#         while True:
#             if ser.in_waiting > 0:  #
#                 data = ser.readline().decode('utf-8').strip()  # ?????? ?��? ?????
#                 print(f"Received: {data}")
#                 ser.write("0\r".encode('utf-8'))
#                 break
#     except KeyboardInterrupt:
#         print("\nExiting...")
#     finally:
#         ser.close()  # ???? ?? ??? ???

# if __name__ == "__main__":
#     while 1:
       

import time
import numpy as np
import tensorflow as tf
from gpiozero import DistanceSensor
import cv2
from collections import Counter
import serial

# 하드웨어 설정
ECHO_PIN = 23
TRIGGER_PIN = 24
ROI_SIZE = 200
MODEL_PATH = "digit_recognition_model.h5"

# 거리 임계값 (50cm 이하일 때 선박 접근으로 판단)
DISTANCE_THRESHOLD = 50

# 숫자 인식 횟수
RECOGNITION_COUNT = 10

# UART 포트 설정
# S32K144 요청 수신용: /dev/ttyAMA2
# 여기서는 S32K144로부터 BLANK 요청을 받는다고 가정.
REQUEST_UART_PORT = "/dev/ttyAMA2"
REQUEST_BAUDRATE = 9600

DATA_UART_PORT = "/dev/ttyAMA2"
DATA_BAUDRATE = 9600

#초음파 거리 인식 함수수
def check_distance(threshold=DISTANCE_THRESHOLD):
    sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN)
    distance_cm = sensor.distance * 100
    print(f"Distance: {distance_cm:.2f} cm")
    return distance_cm <= threshold
#----------------------숫자 인식 관련 함수----------------------
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype('float32') / 255.0
    gray = gray.reshape(1, 28, 28, 1)
    return gray

def recognize_digit(frame, model):
    height, width = frame.shape[:2]
    x_center, y_center = width // 2, height // 2
    x_start = x_center - ROI_SIZE // 2
    y_start = y_center - ROI_SIZE // 2
    roi = frame[y_start:y_start + ROI_SIZE, x_start:x_start + ROI_SIZE]
    cv2.imshow("ROI", roi)
    cv2.waitKey(1)

    preprocessed_roi = preprocess_image(roi)
    predictions = model.predict(preprocessed_roi)
    predicted_label = np.argmax(predictions)
    return predicted_label
#UART 송신 함수
def send_to_uart(uart, data):
    uart.write((data+"\n").encode('utf-8'))
    print(f"Sent to UART: {data}")

def main():
    # 모델 로드
    model = tf.keras.models.load_model(MODEL_PATH)

    # UART 초기화 (요청 수신 및 데이터 전송)
    uart = serial.Serial(port=REQUEST_UART_PORT, baudrate=REQUEST_BAUDRATE, timeout=1)

    # 카메라 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return

    print("System running. Waiting for S32K144 requests...")

    try:
        while True:
            # 1. S32K144의 요청(BLANK) 수신 대기
            # BLANK 문자를 보내온다고 가정하므로, in_waiting > 0일 때 read
            if uart.in_waiting > 0:
                data = uart.readline().decode('utf-8').strip()
                if data == 'B':  # BLANK 요청 가정
                    print("Received request from S32K144.")
                    # 2. 초음파 센서로 선박 존재 확인
                    if not check_distance():
                        # 선박 없음
                        send_to_uart(uart, "NOT_EXIST")
                    else:
                        # 선박 있음 → 숫자 인식
                        print("Object detected! Starting digit recognition.")
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
                        
                        if recognized_digits:
                            most_frequent_digit = Counter(recognized_digits).most_common(1)[0][0]
                            print(f"Most frequently recognized digit: {most_frequent_digit}")

                            # 인식된 숫자 전송
                            send_to_uart(uart, str(most_frequent_digit))
                        else:
                            # 인식 실패 시 NOT_EXIST 대체 가능
                            print("No digits recognized. Sending NOT_EXIST.")
                            send_to_uart(uart, "NOT_EXIST")

                else:
                    # 요청 형식이 BLANK('B')가 아닐 경우 처리 로직 (필요하면 작성)
                    print(f"Unknown request: {data}")
            else:
                # 요청 없으면 잠시 대기
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        uart.close()

if __name__ == "__main__":
    main()

