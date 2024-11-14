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
    
    def get_all_sleep_positions(self, userId: str, expireAfterSeconds: float=180):
        # Tính toán thời gian hiện tại và thời gian cách đây 180 giây
        now = datetime.now(timezone.utc)  # Lấy thời gian UTC hiện tại
        time_threshold = now - timedelta(seconds=expireAfterSeconds)  # Tính thời gian ngưỡng

        try:
            # Truy vấn MongoDB để lấy tất cả các tài liệu của userId có timestamp trong 180 giây qua
            documents = list(self.collection.find(
                {
                    "userId": userId,
                    "timestamp": {"$gte": time_threshold}  # Điều kiện: timestamp >= thời gian ngưỡng
                }
            ))

            # Chuyển ObjectId thành chuỗi để dễ dàng sử dụng
            for doc in documents:
                doc['_id'] = str(doc['_id'])

            return documents  # Trả về các tài liệu phù hợp
        except Exception as e:
            print("Failed to retrieve documents:", e)
            return []  # Trả về danh sách rỗng nếu gặp lỗi
        
    def delete_all_sleep_positions_by_userId(self, userId: str):
        try:
            self.collection.delete_many({"userId": userId})
            print("Documents deleted successfully.")
        except Exception as e:
            print("Document deletion failed:", e)