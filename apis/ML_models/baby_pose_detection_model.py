import os
import pickle
import mediapipe as mp
import cv2
import copy
import pandas as pd
import numpy as np
from PIL import Image

from ML_models.utils.pose_rotation_helper import PoseRotationHelper
from ML_models.utils.pose_scaler_helper import PoseScalerHelper

class BabyPoseDetectionModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.current_path = os.path.dirname(os.path.realpath(__file__))
        self.model = self.load_model(f"{self.current_path}\\best_models\\svc.pkl")
        self.input_scaler = self.load_model(f"{self.current_path}\\best_models\\input_scaler.pkl")
        self.IMPORTANT_LMS = [
            "nose",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_pinky",
            "right_pinky",
            "left_index",
            "right_index",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_foot_index",
            "right_foot_index",
        ]
        self.pose_rotation_helper = PoseRotationHelper()
        self.pose_scaler_helper = PoseScalerHelper(self.IMPORTANT_LMS)

    def load_model(self, file_name):
        with open(file_name, "rb") as file:
            model = pickle.load(file)
        return model
    
    def extract_and_recalculate_landmarks(self, pose_landmarks):
        """
        Tịnh tiến thân người vào giữa bức hình, đồng thời dời lại trục toạ độ
        """
        columns_name = []
        columns_value = []
        for id, landmark in enumerate(pose_landmarks):
            land_mark_name = mp.solutions.pose.PoseLandmark(id).name.lower()
            if land_mark_name not in self.IMPORTANT_LMS:
                continue
            
            columns_name += [
                f"{ land_mark_name }_x",
                f"{ land_mark_name }_y",
                f"{ land_mark_name }_z",
            ]

            # landmark.x, landmark.y là các giá trị trước khi dịch chuyển gốc toạ độ vào giữa bức hình
            # Do đó khi đưa gốc toạ độ về giữa bức hình thì phải trừ chúng cho 0.5
            columns_value += [
                landmark.x - 0.5,
                landmark.y - 0.5,
                landmark.z,
            ]

        df_key_points = pd.DataFrame([columns_value], columns=columns_name)

        # Lấy tọa độ hông trái và phải
        left_hip = (df_key_points["left_hip_x"], df_key_points["left_hip_y"])
        right_hip = (df_key_points["right_hip_x"], df_key_points["right_hip_y"])

        # Tìm điểm trung tâm của hông
        center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Lấy tọa độ của mũi
        nose = (df_key_points["nose_x"], df_key_points["nose_y"])

        # Khoảng cách giữa mũi và trung tâm hông
        distance = np.sqrt((center_hip[0] - nose[0])**2 + (center_hip[1] - nose[1])**2)

        # Tính toán scale value
        scale_value = 0.5 / distance

        # **Scale** tất cả các key points
        for column in df_key_points.columns:
            if "_x" in column or "_y" in column:
                df_key_points[column] = df_key_points[column] * scale_value

        return df_key_points
    
    def predict(self, image, prediction_probability_threshold=0.5) -> int:
        with self.mp_pose.Pose(
            static_image_mode=True, model_complexity=1, smooth_landmarks=True
        ) as pose:
            
            image, new_size = self.process_image(image)
            results = pose.process(image)

            if not results.pose_landmarks:
                raise Exception("No pose landmarks detected")

            image.flags.writeable = True

            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get landmarks
            try:
                results.pose_landmarks = self.pose_rotation_helper.rotate_keypoints(results.pose_landmarks, new_size)

                # draw_landmarks(mp_drawing, mp_pose, image, results.pose_landmarks)

                key_points_df = self.pose_scaler_helper.extract_and_recalculate_landmarks(results.pose_landmarks.landmark)

                # Convert DataFrame to numpy array
                key_points = key_points_df.values.reshape(1, -1)  # Chuyển DataFrame thành mảng 2D với đúng kích thước

                # Scale input trước khi dự đoán
                X = self.input_scaler.transform(key_points)

                # Dự đoán
                predicted_class = self.model.predict(X)[0]  # Dự đoán dựa trên mô hình và input đã được scale

                return int(predicted_class)
            except Exception as e:
                raise Exception(f"Error when predicting: {e}")
    
    def square_for_image(self, image: np.array):
        # Nếu hình ảnh là một đối tượng NumPy (cv2 image)
        if isinstance(image, np.ndarray):
            # Chuyển đổi từ định dạng cv2 (NumPy array) sang PIL image để xử lý
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            # Nếu hình ảnh đã ở định dạng PIL thì không cần chuyển đổi
            image_pil = image

        # Lấy kích thước ảnh gốc
        original_size = image_pil.size

        # Kích thước mới sẽ là kích thước lớn nhất giữa chiều rộng và chiều cao
        max_width = max(original_size)
        new_size = (max_width, max_width)

        # Tạo một ảnh mới với nền đen (kích thước vuông)
        new_image = Image.new("RGB", new_size, (0, 0, 0))

        # Dán ảnh gốc vào ảnh mới với khoảng trống màu đen
        new_image.paste(image_pil, 
                        ((max_width - original_size[0]) // 2, (max_width - original_size[1]) // 2))

        # Chuyển ảnh mới từ định dạng PIL sang định dạng NumPy (cv2)
        new_image_cv2 = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)

        return new_image_cv2, new_size

    def process_image(self, image):
        """Load and pre-process the image."""
        image, new_size = self.square_for_image(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), new_size