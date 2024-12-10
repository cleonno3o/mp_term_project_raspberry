from flask import Flask, Response
import cv2
import threading

app = Flask(__name__)

cap = cv2.VideoCapture(0)
lock = threading.Lock()  # 동기화 Lock

def generate_frames():
    global cap
    while True:
        with lock:  # 동기화
            success, frame = cap.read()
        if not success:
            break
        else:

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=33389)
    finally:

        cap.release()
