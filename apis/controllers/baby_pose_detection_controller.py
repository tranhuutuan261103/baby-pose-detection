from flask import Blueprint, jsonify, request
import cv2
import numpy as np
from services.baby_pose_detection_service import BabyPoseDetectionService

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

        image_file = request.files["image"]

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
        return jsonify(result)

    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500