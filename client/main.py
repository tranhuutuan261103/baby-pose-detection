import cv2
import matplotlib.pyplot as plt
import requests
import numpy as np
import threading

def call_api(image):
    global current_class  # Khai báo biến global trước khi sử dụng
    # Encode the frame as a JPEG image
    _, img_encoded = cv2.imencode('.jpg', image)
    img_bytes = img_encoded.tobytes()

    # Create a multipart/form-data POST request
    files = {
        'image': ('frame.jpg', img_bytes, 'image/jpeg')
    }
    response = requests.post("http://localhost:6666/api/baby_pose_detection/predict", files=files)

    # Lấy message_vn từ response
    current_class = response.json().get("message_vn", "Unknown")

def call_api_in_thread(image):
    # Tạo thread để thực hiện hàm call_api
    api_thread = threading.Thread(target=call_api, args=(image,))
    api_thread.start()

current_class = None  # Khởi tạo biến toàn cục

if __name__ == "__main__":
    cap = cv2.VideoCapture("./client/data_test/full.mp4")

    if not cap.isOpened():
        print("Error opening video stream or file")

    i = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Call API every 30 frames
        if i % 30 == 0:
            call_api_in_thread(frame)

        i += 1

        if current_class is not None:
            plt.text(0, -10, current_class, fontsize=16, color='green')

        # Sử dụng matplotlib để hiển thị frame
        plt.imshow(frame_rgb)
        plt.pause(0.001)  # Dừng ngắn để cập nhật plot
        plt.clf()  # Xóa plot để hiển thị frame tiếp theo

    cap.release()
