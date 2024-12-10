import subprocess
from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

cap = None
lock = threading.Lock()  # 동기화 Lock
selected_camera = 0  # 기본 카메라 인덱스


def list_available_cameras():
    """macOS에서 사용 가능한 카메라 이름과 인덱스를 가져옴"""
    try:
        # ffmpeg 명령어 실행
        cmd = ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", "dummy"]
        result = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        output = result.stderr

        # 카메라 정보 파싱
        cameras = []
        for line in output.splitlines():
            if "[AVFoundation input device @ " in line and "video devices" in line:
                camera_name = line.split("] ")[-1]
                index = len(cameras)  # 인덱스는 0부터 시작
                cameras.append((index, camera_name))

        return cameras
    except Exception as e:
        print(f"[ERROR] Failed to list cameras: {e}")
        return []


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
    """프레임 생성"""
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
    for idx, name in available_cameras:
        print(f" - Camera {idx}: {name}")

    while True:
        try:
            selected = int(input("Enter the camera index to use: "))
            if any(selected == cam[0] for cam in available_cameras):
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
