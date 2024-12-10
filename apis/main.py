import os
import cv2
import base64
import threading
import numpy as np
from time import time
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import controllers
from controllers.baby_in_crib_detection_controller import bicd_bp
from controllers.baby_pose_detection_controller import bpd_bp
from controllers.baby_sleep_position_controller import bsp_bp
from controllers.baby_sleep_position_history_controller import bsph_bp
from controllers.infant_cry_classification_controller import icc_bp

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
CORS(app)
socketio = SocketIO(app)

# Video storage folder
video_folder = "apis/media/videos/"
if not os.path.exists(video_folder):
    os.makedirs(video_folder)

# User-specific data
video_writers = {}       # {user_id: VideoWriter}
video_frames = {}        # {user_id: []}
recording_states = {}    # {user_id: bool}
last_time = {}           # {user_id: float (timestamp)}
locks = {}               # {user_id: threading.Lock}

FRAME_INTERVAL = 1 / 30  # 30 FPS

# Helper Functions
def start_video_recording(user_id):
    """Starts video recording for a specific user."""
    global video_writers, recording_states, locks

    # Ensure lock for the user
    if user_id not in locks:
        locks[user_id] = threading.Lock()

    with locks[user_id]:
        recording_states[user_id] = True
        video_filename = os.path.join(video_folder, f"{user_id}_video.avi")

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writers[user_id] = cv2.VideoWriter(video_filename, fourcc, 30.0, (640, 480))

        if not video_writers[user_id].isOpened():
            print(f"Error: Could not open VideoWriter for user {user_id}.")
            recording_states[user_id] = False
            return

        video_frames[user_id] = []
        last_time[user_id] = time()  # Initialize the last frame time
        print(f"Recording started for user {user_id}.")

def save_video(user_id):
    """Saves video for a specific user."""
    global video_writers, locks

    with locks[user_id]:
        if user_id in video_writers and video_writers[user_id]:
            video_writers[user_id].release()
            print(f"Video saved for user {user_id}.")
        
        reset_video_recording(user_id)

def reset_video_recording(user_id):
    """Resets recording state for a specific user."""
    global recording_states, video_writers, video_frames, locks

    with locks[user_id]:
        recording_states[user_id] = False
        video_writers.pop(user_id, None)
        video_frames.pop(user_id, None)
        last_time.pop(user_id, None)

def handle_video_data(data, user_id):
    """Processes incoming video data for a specific user."""
    global video_writers, video_frames, last_time, locks

    if 'image' not in data or not data['image']:
        print(f"Error: No image data received for user {user_id}.")
        return

    image_data = str(data['image'])

    if image_data.startswith("data:image/jpeg;base64,"):
        image_data = image_data.split(',')[1]

    try:
        img_data = base64.b64decode(image_data)
        image = Image.open(BytesIO(img_data))
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        current_time = time()
        with locks[user_id]:
            # Only process the frame if sufficient time has passed
            if current_time - last_time.get(user_id, 0) >= FRAME_INTERVAL:
                last_time[user_id] = current_time

                # Add frame to user's video frames
                if user_id not in video_frames:
                    video_frames[user_id] = []
                video_frames[user_id].append(frame)

                # Write frame to video if recording
                if recording_states.get(user_id, False) and user_id in video_writers:
                    video_writers[user_id].write(frame)

    except Exception as e:
        print(f"Error processing video frame for user {user_id}: {e}")

# SocketIO event listeners
@socketio.on('start_recording')
def handle_start_recording(data: dict):
    user_id = data.get('user_id', 'unknown')
    print(f"Received start_recording event for user {user_id}.")
    start_video_recording(user_id)

@socketio.on('video')
def handle_video(data):
    user_id = data.get('user_id', 'unknown')
    handle_video_data(data, user_id)

@socketio.on('stop_recording')
def handle_stop_recording(data):
    user_id = data.get('user_id', 'unknown')
    print(f"Received stop_recording event for user {user_id}.")
    save_video(user_id)

@socketio.on('connect')
def handle_connect():
    print("Client connected")
    socketio.emit('response', {'message': 'Server: Connection established!'})

# Main entry point
if __name__ == '__main__':
    app.register_blueprint(bicd_bp)
    app.register_blueprint(bpd_bp)
    app.register_blueprint(bsp_bp)
    app.register_blueprint(bsph_bp)
    app.register_blueprint(icc_bp)
    socketio.run(app, host="0.0.0.0", port=4656, debug=True)