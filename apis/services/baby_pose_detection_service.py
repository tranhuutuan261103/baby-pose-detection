import cv2
from ML_models.baby_pose_detection_model import BabyPoseDetectionModel

class BabyPoseDetectionService:
    def __init__(self):
        self.model = BabyPoseDetectionModel()

    def predict(self, image) -> dict:
        try:
            result = self.model.predict(image)
            if result == 0:
                return {
                    "id": 0,
                    "message": "Baby is lying on",
                    "message_vn": "Trẻ đang nằm ngửa"
                }
            elif result == 1:
                return {
                    "id": 1,
                    "message": "Baby is lying on one side",
                    "message_vn": "Trẻ đang nằm nghiêng về một bên"
                }
            elif result == 2:
                return {
                    "id": 2,
                    "message": "Baby is lying on his stomach",
                    "message_vn": "Trẻ đang nằm sấp"
                }
        except Exception as e:
            raise e