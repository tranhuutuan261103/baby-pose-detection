import cv2
from ML_models.baby_pose_detection_model import BabyPoseDetectionModel

class BabyPoseDetectionService:
    def __init__(self):
        self.model = BabyPoseDetectionModel()

    def predict(self, image) -> str:
        try:
            return self.model.predict(image)
        except Exception as e:
            raise Exception(f"Prediction failed: {str(e)}")