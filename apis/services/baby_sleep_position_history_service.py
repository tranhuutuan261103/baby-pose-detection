import pymongo
from datetime import datetime, timedelta, timezone
from services.mongodb_helper import db

class BabySleepPositionHistoryService:
    def __init__(self):
        self.collection_name = "baby_sleep_positions_history"
        self.collection = db[self.collection_name]
    
    def insert_sleep_position(self, data):
        try:
            self.collection.insert_one(data)
            print("Documents inserted successfully.")
        except Exception as e:
            print("Document insertion failed:", e)
    
    def get_all_sleep_positions(self, userId: str, expireAfterSeconds: float=180 * 100):
        now = datetime.now(timezone.utc)  # Lấy thời gian UTC hiện tại
        time_threshold = now - timedelta(seconds=expireAfterSeconds)

        try:
            documents = list(self.collection.find(
                {
                    "userId": userId,
                    "timestamp": {"$gte": time_threshold}
                }
            ))

            for doc in documents:
                doc['_id'] = str(doc['_id'])
                
                # Chuyển đổi timestamp thành timezone-aware nếu nó không có múi giờ
                if doc.get("timestamp") and doc["timestamp"].tzinfo is None:
                    doc["timestamp"] = doc["timestamp"].replace(tzinfo=timezone.utc)

            return documents
        except Exception as e:
            print("Failed to retrieve documents:", e)
            return []
        
    def delete_all_sleep_positions_by_userId(self, userId: str):
        try:
            self.collection.delete_many({"userId": userId})
            print("Documents deleted successfully.")
        except Exception as e:
            print("Document deletion failed:", e)