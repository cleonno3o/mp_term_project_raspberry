import cv2
import numpy as np
import tensorflow as tf

# 학습된 모델 로드
model_path = "digit_recognition_model.h5"  # 생성한 모델 파일 경로
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

# 웹캠 초기화
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the webcam.")
        break

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

    # 결과 출력
    cv2.imshow("Webcam - Digit Recognition", frame)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
