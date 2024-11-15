from flask import Blueprint, jsonify, request
import cv2
import numpy as np
from datetime import datetime, timezone, timedelta
from services.baby_cry_adult_voice_classification_service import BabyCryAdultVoiceClassificationService
from services.infant_cry_classification_service import InfantCryClassificationService
from services.firebase_helper import save_file_to_firestore, get_account_info_by_id, save_log_to_firestore, save_notification_to_firebase
from services.message_helper import send_notification_to_device
from services.utils import most_frequent_element
import logging
# Configure the logger
logging.basicConfig(filename='apis/server_logs.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

import os

audio_folder = "apis/media/audios/"
if not os.path.exists(audio_folder):
    os.makedirs(audio_folder)

icc_bp = Blueprint("infant_cry_classification", __name__, url_prefix="/api/infant_cry_classification")
babyCryAdultVoiceClassificationService = BabyCryAdultVoiceClassificationService()
infantCryClassificationService = InfantCryClassificationService()
    
@icc_bp.route("/predict", methods=["POST"])
def predict_infant_cry():
    try:
        # Log request information for debugging
        if "audio" not in request.files:
            return jsonify({"message": "No audio part in the request"}), 400
        
        if "system_id" not in request.form:
            return jsonify({"message": "No system_id part in the request"}), 400

        audio_file = request.files["audio"]

        system_id = request.form["system_id"]

        # Get account info by code
        account_info = get_account_info_by_id(system_id)
        if account_info is None:
            return jsonify({"message": "No account found with code"}), 400

        # save file to disk
        audio_file_name = f"{system_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        audio_file.save(os.path.join(audio_folder, audio_file_name))
        save_file_to_firestore(os.path.join(audio_folder, audio_file_name), f"{system_id}_audio.wav")

        # Call the model's prediction function
        result = babyCryAdultVoiceClassificationService.predict(os.path.join(audio_folder, audio_file_name))

        if result == 0:
            save_log_to_firestore("audio", audio_file_name, "Trẻ bình thường", system_id, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))
        elif result == 1:
            save_log_to_firestore("audio", audio_file_name, "Trẻ đang khóc", system_id, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))

        if result == 1:
            logging.info(f"Send notification to device {account_info['deviceToken']}")
            send_notification_to_device(account_info["deviceToken"], "Trẻ đang khóc", "Trẻ đang khóc")
            save_notification_to_firebase("Trẻ đang khóc", system_id, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))
            cry_classes = infantCryClassificationService.predict(os.path.join(audio_folder, audio_file_name))
            if len(cry_classes) == 0:
                return jsonify(
                    {
                        "result": str(result),
                        "type": "Không phát hiện tiếng khóc"
                    }
                )
            
            cry_class = most_frequent_element(cry_classes)
            if account_info["enableNotification"] == True:
                send_notification_to_device(account_info["deviceToken"], "Thông báo từ hệ thống", f"{cry_class}")
            return jsonify(
                {
                    "result": str(result),
                    "type": str(cry_class)
                }
            )

        return jsonify(
            {
                "result": str(result)
            }
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500