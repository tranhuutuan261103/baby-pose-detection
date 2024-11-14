from flask import Blueprint, jsonify, request
import cv2
import numpy as np
from datetime import datetime
from services.baby_cry_adult_voice_classification_service import BabyCryAdultVoiceClassificationService

import os

audio_folder = "apis/media/audio/"
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

icc_bp = Blueprint("infant_cry_classification", __name__, url_prefix="/api/infant_cry_classification")
babyCryAdultVoiceClassificationService = BabyCryAdultVoiceClassificationService()
    
@icc_bp.route("/predict", methods=["POST"])
def predict_infant_cry():
    try:
        # Log request information for debugging
        if "audio" not in request.files:
            return jsonify({"message": "No audio part in the request"}), 400

        audio_file = request.files["audio"]

        # save file to disk
        audio_file.save(os.path.join(audio_folder, audio_file.filename))

        # Call the model's prediction function
        result = babyCryAdultVoiceClassificationService.predict(os.path.join(audio_folder, audio_file.filename))

        return jsonify(
            {
                "result": str(result)
            }
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500