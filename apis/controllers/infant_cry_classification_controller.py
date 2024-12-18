import logging

from flask import Blueprint, jsonify, request
from datetime import datetime, timezone, timedelta
from services.AI.baby_cry_adult_voice_classification_service import BabyCryAdultVoiceClassificationService
from services.AI.infant_cry_classification_service import InfantCryClassificationService
from services.firebase_helper import save_file_to_firestore, get_account_infos_by_id, data_observer, save_log_to_firestore, save_notification_to_firebase
from services.notification_service import send_notification_to_user

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
        account_infos = get_account_infos_by_id(system_id)
        if len(account_infos) == 0:
            return jsonify({"message": "No account found with code"}), 400

        # save file to disk
        audio_file_name = f"{system_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
        audio_file.save(os.path.join(audio_folder, audio_file_name))
        save_file_to_firestore(os.path.join(audio_folder, audio_file_name), f"{system_id}_audio.wav")
        data_observer(f"data_observer/{system_id}/is_updated_audio", True)

        # Call the model's prediction function
        result = babyCryAdultVoiceClassificationService.predict(os.path.join(audio_folder, audio_file_name))

        if result == 0:
            save_log_to_firestore("audio", audio_file_name, "Trẻ bình thường", system_id, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))
        elif result == 1:
            save_log_to_firestore("audio", audio_file_name, "Trẻ đang khóc", system_id, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))

        if result == 1:
            for account_info in account_infos:
                logging.info(f"Send notification to device {account_info['deviceToken']}")
                send_notification_to_user(account_info, "Trẻ đang khóc", "Phát hiện trẻ đang khóc")
            
            # cry_class = infantCryClassificationService.predict(os.path.join(audio_folder, audio_file_name))
            # if cry_class == None:
            #     return jsonify(
            #         {
            #             "result": str(result)
            #         }
            #     )
            
            # if account_infos[0]["enableNotification"] == True:
            #     class_name = "Trẻ đang khóc"
            #     if cry_class == 0:
            #         class_name = "Trẻ cảm thấy đau bụng"
            #     elif cry_class == 1:
            #         class_name = "Trẻ cảm thấy ợ hơi"
            #     elif cry_class == 2:
            #         class_name = "Trẻ cảm thấy không thoải mái"
            #     elif cry_class == 3:
            #         class_name = "Trẻ cảm thấy lo sợ"
            #     elif cry_class == 4:
            #         class_name = "Trẻ cảm thấy mệt mỏi"
                
            #     for account_info in account_infos:
            #         send_notification_to_user(account_info, "Trẻ đang khóc", class_name)
            #     return jsonify(
            #         {
            #             "result": str(result),
            #             "type": str(class_name)
            #         }
            #     )

        return jsonify(
            {
                "result": str(result)
            }
        )
    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500