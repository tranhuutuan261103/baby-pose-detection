from flask import Blueprint, jsonify, request
from datetime import datetime, timezone, timedelta
from services.baby_sleep_position_history_service import BabySleepPositionHistoryService

bsph_bp = Blueprint("baby_sleep_position_history", __name__, url_prefix="/api/baby_sleep_position_history")
babySleepPositionHistoryService = BabySleepPositionHistoryService()
    
@bsph_bp.route("", methods=["GET"])
def get_all_sleep_positions():
    try:
        userId = request.args.get('userId')
        return jsonify(babySleepPositionHistoryService.get_all_sleep_positions(userId))
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})

@bsph_bp.route("check", methods=["GET"])
def get_sleep_positions_by_date():
    try:
        lately_sleep_positions_history = babySleepPositionHistoryService.get_all_sleep_positions("pbl6_01")
        print(lately_sleep_positions_history)
        if len(lately_sleep_positions_history) > 0:
            if lately_sleep_positions_history[0]["timestamp"] <= datetime.now(timezone.utc) - timedelta(minutes=4):
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
                    return jsonify({"message": "Ngửa"})
                elif count_position_1 > count_position_0 * 2:
                    return jsonify({"message": "Nghiêng"})
                else:
                    return jsonify({"message": "Bình thường"})
            else:
                return jsonify({"message": "Bình thường"})
        else:
            return jsonify({"message": "Bình thường"})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})
    
@bsph_bp.route("/insert", methods=["POST"])
def insert_sleep_positions():
    try:
        data = request.get_json()
        now = datetime.now(timezone.utc)  # Lấy thời gian UTC hiện tại
        data["timestamp"] = now
        babySleepPositionHistoryService.insert_sleep_position(data)
        return jsonify({"message": "Data inserted successfully"})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})