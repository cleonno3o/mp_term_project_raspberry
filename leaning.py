import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# **1. 데이터 로드 및 전처리**
def load_data(data_dir):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, label)
        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                # 이미지 읽기 (Grayscale)
                image = tf.keras.preprocessing.image.load_img(image_path, color_mode="grayscale", target_size=(28, 28))
                image = tf.keras.preprocessing.image.img_to_array(image)
                images.append(image)
                labels.append(int(label))  # 폴더 이름이 라벨로 사용됨
    return np.array(images), np.array(labels)

# 데이터 로드
data_dir = "assets"  # 'assets' 폴더 경로 입력
X, y = load_data(data_dir)

# 데이터 정규화 및 전처리
X = X / 255.0  # 정규화
y = to_categorical(y, num_classes=11)  # 라벨을 원-핫 인코딩 (1~10까지 숫자, 0 포함)

# 데이터 분할 (훈련:검증 = 8:2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **2. 모델 설계**
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(11, activation='softmax')  # 11개의 클래스 (0~10)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# **3. 학습**
history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_val, y_val))

# **4. 모델 저장**
model.save("digit_recognition_model.h5")
print("Model saved as 'digit_recognition_model.h5'")

# **5. 학습 결과 시각화 (선택 사항)**
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")
plt.show()
