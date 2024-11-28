from flask import Blueprint, jsonify, request
import cv2
import os
import numpy as np

from services.baby_in_crib_detection_service import BabyInCribDetectionService

bicd_bp = Blueprint("baby_in_crib_detection", __name__, url_prefix="/api/baby_in_crib_detection")

babyInCribDetectionService = BabyInCribDetectionService()

@bicd_bp.route("/predict", methods=["POST"])
def predict_baby_in_crib_detection():
    try:
        if "image" not in request.files:
            return jsonify({"message": "No image part in the request"}), 400
        
        # if "system_id" not in request.form:
        #     return jsonify({"message": "No system_id part in the request"}), 400

        image_file = request.files["image"]
        # code = request.form["system_id"]

        if image_file.filename == '':
            return jsonify({"message": "No image selected for uploading"}), 400
        
        # Read and process the image file
        image_bytes = np.frombuffer(image_file.stream.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Predict
        result = babyInCribDetectionService.predict(image)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})