import time
import numpy as np
import tensorflow as tf
from gpiozero import DistanceSensor
import cv2
from collections import Counter
import serial
from flask import Flask, Response
import threading

# Flask 설정
app = Flask(__name__)

# 하드웨어 및 설정
ECHO_PIN = 23
TRIGGER_PIN = 24
ROI_SIZE = 200
MODEL_PATH = "digit_recognition_model.h5"

DISTANCE_THRESHOLD = 50
RECOGNITION_COUNT = 10

REQUEST_UART_PORT = "/dev/ttyAMA2"
REQUEST_BAUDRATE = 9600

# 모델 로드
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# 거리 센서
def check_distance(threshold=DISTANCE_THRESHOLD):
    sensor = DistanceSensor(echo=ECHO_PIN, trigger=TRIGGER_PIN)
    distance_cm = sensor.distance * 100
    print(f"Distance: {distance_cm:.2f} cm")
    return distance_cm <= threshold

# 이미지 전처리
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (28, 28))
    gray = gray.astype('float32') / 255.0
    gray = gray.reshape(1, 28, 28, 1)
    return gray

# 숫자 인식
def recognize_digit(frame):
    height, width = frame.shape[:2]
    x_center, y_center = width // 2, height // 2
    x_start = x_center - ROI_SIZE // 2
    y_start = y_center - ROI_SIZE // 2
    roi = frame[y_start:y_start + ROI_SIZE, x_start:x_start + ROI_SIZE]
    preprocessed_roi = preprocess_image(roi)
    predictions = model.predict(preprocessed_roi)
    return np.argmax(predictions), roi

# Flask 비디오 스트림
cap = None
lock = threading.Lock()
streaming = False

def generate_frames():
    """Flask용 비디오 스트림"""
    global cap, streaming
    while streaming:
        with lock:
            if cap is None:
                break
            success, frame = cap.read()
        if not success:
            break
        else:
            recognized_digit, roi = recognize_digit(frame)
            label = f"Recognized Digit: {recognized_digit}"
            cv2.rectangle(frame, (ROI_SIZE//2, ROI_SIZE//2), 
                          (ROI_SIZE, ROI_SIZE), (0, 255, 0), 2)
            cv2.putText(frame, label, (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video')
def video_feed():
    """웹 비디오 스트림"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# UART 통신 및 메인 로직
def main():
    global cap, streaming
    uart = serial.Serial(port=REQUEST_UART_PORT, baudrate=REQUEST_BAUDRATE, timeout=1)
    print("System running. Waiting for S32K144 requests...")

    try:
        while True:
            if uart.in_waiting > 0:
                data = uart.readline().decode('utf-8').strip()
                if data == 'B':  # 요청이 들어옴
                    print("Received request from S32K144.")

                    # 거리 확인
                    if not check_distance():
                        uart.write(b"NOT_EXIST\r\n")
                    else:
                        print("Object detected! Starting digit recognition.")

                        # 카메라 시작 및 Flask 비디오 활성화
                        with lock:
                            cap = cv2.VideoCapture(0)
                            streaming = True

                        # 인식된 숫자 저장
                        recognized_digits = []
                        for _ in range(RECOGNITION_COUNT):
                            with lock:
                                success, frame = cap.read()
                            if success:
                                digit, _ = recognize_digit(frame)
                                recognized_digits.append(digit)
                            time.sleep(0.5)

                        # 결과 송신
                        if recognized_digits:
                            most_frequent_digit = Counter(recognized_digits).most_common(1)[0][0]
                            uart.write(f"{most_frequent_digit}\r\n".encode())
                        else:
                            uart.write(b"NOT_EXIST\r\n")

                        # 카메라 종료
                        with lock:
                            cap.release()
                            cap = None
                            streaming = False
                else:
                    time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        if cap:
            cap.release()
        uart.close()

# Flask 및 메인 동시 실행
if __name__ == "__main__":
    threading.Thread(target=main, daemon=True).start()
    app.run(host="0.0.0.0", port=33389, debug=True)
