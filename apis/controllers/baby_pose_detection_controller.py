from flask import Blueprint, jsonify, request
import cv2
import os
import numpy as np
from services.AI.baby_pose_detection_service import BabyPoseDetectionService
from services.baby_sleep_position_service import BabySleepPositionService
from services.baby_sleep_position_history_service import BabySleepPositionHistoryService
from services.firebase_helper import get_account_infos_by_id, save_file_to_firestore, data_observer, save_log_to_firestore, save_notification_to_firebase
from services.notification_service import send_notification_to_user
from datetime import datetime, timedelta, timezone

bpd_bp = Blueprint("baby_pose_detection", __name__, url_prefix="/api/baby_pose_detection")
image_folder = "apis/media/images/"
if not os.path.exists(image_folder):
    os.makedirs(image_folder)

babyPoseDetectionService = BabyPoseDetectionService()
babySleepPositionService = BabySleepPositionService()
babySleepPositionHistoryService = BabySleepPositionHistoryService()

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

        # Check if the file is empty
        if image_file.filename == '':
            return jsonify({"message": "No image selected for uploading"}), 400

        # Get account info by code
        account_infos = get_account_infos_by_id(code)
        if len(account_infos) == 0:
            return jsonify({"message": "No account found with code"}), 400

        # Read and process the image file
        image_bytes = np.frombuffer(image_file.stream.read(), np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        # Check if image was successfully decoded
        if image is None:
            return jsonify({"message": "Image decoding failed"}), 400
        
        # Đặt tên file tạm dựa trên ID và timestamp để tránh trùng lặp
        temp_image_name = f"{code}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
        cv2.imwrite(os.path.join(image_folder, temp_image_name), image)

        # Tải ảnh lên Firebase từ file tạm
        image_url = save_file_to_firestore(os.path.join(image_folder, temp_image_name), f"{code}_image.jpg")
        if image_url is None:
            print("Error saving image to Firestore")
        data_observer(f"data_observer/{code}/is_updated_image", True)

        # Call the model's prediction function
        result = babyPoseDetectionService.predict(image)

        # Save the log to Firestore
        class_type = "Unknown"
        if result["id"] == 0:
            class_type = "Nằm ngửa"
        elif result["id"] == 1:
            class_type = "Nằm nghiêng"
        elif result["id"] == 2:
            class_type = "Nằm xấp"

        save_log_to_firestore("image", temp_image_name, class_type, code, (datetime.now(timezone.utc) + timedelta(hours=7)).strftime('%Y-%m-%dT%H:%M:%S.000'))

        if result["id"] == 2:
            # Send notification to user
            for account_info in account_infos:
                if account_info["enableNotification"] == True:
                    print("Sending notification to user...")
                    send_notification_to_user(account_info, "Cảnh báo tư thế ngủ", "Đã phát hiện trẻ em đang nằm xấp. Vui lòng kiểm tra.")
                    babySleepPositionHistoryService.delete_all_sleep_positions_by_userId(code)
            
        if result["id"] == 0 or result["id"] == 1:
            # Insert sleep position data into MongoDB
            babySleepPositionService.insert_sleep_position({
                "userId": code,
                "timestamp": datetime.now(timezone.utc),
                "positionType": result["id"]
            })

            lately_sleep_positions = babySleepPositionService.get_all_sleep_positions(code)

            is_changed = True
            if len(lately_sleep_positions) > 0:
                for item in lately_sleep_positions:
                    if item["positionType"] != result["id"]:
                        is_changed = False

                if is_changed:
                    babySleepPositionHistoryService.insert_sleep_position({
                        "userId": code,
                        "timestamp": lately_sleep_positions[0]["timestamp"],
                        "positionType": result["id"]
                    })

        if account_infos[0]["enableNotification"] == True:
            lately_sleep_positions_history = babySleepPositionHistoryService.get_all_sleep_positions(code)
            if len(lately_sleep_positions_history) > 0:
                if lately_sleep_positions_history[0]["timestamp"] <= datetime.now(timezone.utc) - timedelta(minutes=account_info["schedule"]):
                    # Calculate the time positionType = 0 and 1 in the last 30 minutes
                    count_position_0 = 0 #s
                    count_position_1 = 0 #s
                    for i in range(len(lately_sleep_positions_history) - 1):
                        if lately_sleep_positions_history[i]["positionType"] == 0:
                            count_position_0 += (lately_sleep_positions_history[i + 1]["timestamp"] - lately_sleep_positions_history[i]["timestamp"]).total_seconds()
                        else:
                            count_position_1 += (lately_sleep_positions_history[i + 1]["timestamp"] - lately_sleep_positions_history[i]["timestamp"]).total_seconds()
                    
                    if lately_sleep_positions_history[-1]["positionType"] == 0:
                        count_position_0 += (datetime.now(timezone.utc) - lately_sleep_positions_history[-1]["timestamp"]).total_seconds()
                    else:
                        count_position_1 += (datetime.now(timezone.utc) - lately_sleep_positions_history[-1]["timestamp"]).total_seconds()

                    if count_position_0 > count_position_1 * 2:
                        for account_info in account_infos:
                            send_notification_to_user(account_info, "Cảnh báo tư thế ngủ", "Trẻ em của bạn đã nằm ngửa quá lâu. Vui lòng kiểm tra.")
                    elif count_position_1 > count_position_0 * 2:
                        for account_info in account_infos:
                            send_notification_to_user(account_info, "Cảnh báo tư thế ngủ", "Trẻ em của bạn đã nằm nghiêng quá lâu. Vui lòng kiểm tra.")
                
                    # delete all sleep positions by userId
                    babySleepPositionHistoryService.delete_all_sleep_positions_by_userId(code)

        return jsonify(result)

    except Exception as e:
        # Log the actual error for debugging
        print(f"Exception: {str(e)}")
        return jsonify({"message": "An error occurred", "error": str(e)}), 500