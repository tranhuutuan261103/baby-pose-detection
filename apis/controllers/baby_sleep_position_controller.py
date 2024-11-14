from flask import Blueprint, jsonify, request
from datetime import datetime, timezone
from services.baby_sleep_position_service import BabySleepPositionService

bsp_bp = Blueprint("baby_sleep_position", __name__, url_prefix="/api/baby_sleep_position")
    
@bsp_bp.route("", methods=["GET"])
def get_all_sleep_positions():
    try:
        userId = request.args.get('userId')
        return jsonify(BabySleepPositionService().get_all_sleep_positions(userId))
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})
    
@bsp_bp.route("/insert", methods=["POST"])
def insert_sleep_positions():
    try:
        data = request.get_json()
        now = datetime.now(timezone.utc)  # Lấy thời gian UTC hiện tại
        data["timestamp"] = now
        BabySleepPositionService().insert_sleep_position(data)
        return jsonify({"message": "Data inserted successfully"})
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)})