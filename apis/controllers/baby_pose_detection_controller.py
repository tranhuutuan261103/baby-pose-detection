from flask import Blueprint, jsonify, request
import cv2
import numpy as np
from services.baby_pose_detection_service import BabyPoseDetectionService
from services.firebase_helper import get_account_info_by_code, send_notification_to_device

bpd_bp = Blueprint("baby_pose_detection", __name__, url_prefix="/api/baby_pose_detection")

@bpd_bp.route("", methods=["GET"])
def get_baby_pose_detection():
    try:
        return jsonify({"message": "This is the baby pose detection API"})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})
    
@bpd_bp.route("/predict", methods=["POST"])
def predict_baby_pose_detection():
    try:
        # Log request information for debugging
        if "image" not in request.files:
            return jsonify({"message": "No image part in the request"}), 400
        
        if "system_id" not in request.form:
            return jsonify({"message": "No system_id part in the request"}), 400

        image_file = request.files["image"]
        code = request.form["system_id"]

        # Get account info by code
        account_info = get_account_info_by_code(code)
        if account_info is None:
            return jsonify({"message": "No account found with code"}), 400

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({"message": "No image selected for uploading"}), 400

        # Read and process the image file
        image_bytes = np.frombuffer(image_file.stream.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Check if image was successfully decoded
        if image is None:
            return jsonify({"message": "Image decoding failed"}), 400

        # Call the model's prediction function
        result = BabyPoseDetectionService().predict(image)

        if account_info["enableNotification"] == True and result["id"] == 2:
            # Send notification to user
            print("Sending notification to user...")
            send_notification_to_device(account_info["deviceToken"], "Thông báo từ hệ thống", "Đã phát hiện trẻ em đang nằm xấp. Vui lòng kiểm tra.")

        return jsonify(result)

    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500