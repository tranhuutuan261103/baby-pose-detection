import os
import pickle
import mediapipe as mp
import cv2
import copy
import pandas as pd
import numpy as np
from ML_models.image_preprocessing import ImagePreprocessing
from PIL import Image

class BabyPoseDetectionModel:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.image_preprocessing = ImagePreprocessing()
        self.model = self.load_model(f"{os.path.dirname(os.path.realpath(__file__))}\\random_forest.pkl")
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
    
    def predict(self, image, prediction_probability_threshold=0.5) -> str:
        current_class = "Unknown"

        with self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        ) as pose:
            
            image, new_size = self.square_for_image(image)

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if not results.pose_landmarks:
                return "No human found"

            initial_pose_landmarks = copy.deepcopy(results.pose_landmarks)
            image.flags.writeable = True

            # Cần khôi phục lại màu gốc của ảnh
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get landmarks
            try:
                print(f"Initial landmarks:")
                results.pose_landmarks = self.rotate_keypoints(results.pose_landmarks, new_size)
                print(f"Rotated landmarks:")
                key_points = self.extract_and_recalculate_landmarks(
                    results.pose_landmarks.landmark
                )
                print(f"Key points: {key_points}")
                # X = pd.DataFrame([key_points], columns=self.HEADERS[1:])
                # X = self.input_scaler.transform(X)

                # print(f"X: {X}")

                predicted_class = self.model.predict(key_points)[0]

                print(f"Predicted class: {predicted_class}")

                return str(predicted_class)

            except Exception as e:
                print(f"Error: {e}")
                return "Prediction failed"
            

    def rotate_keypoints(self, keypoints, origin_size = (612, 408)) -> np.array:
        left_shoulder = keypoints.landmark[11]
        right_shoulder = keypoints.landmark[12]
        left_hip = keypoints.landmark[23]
        right_hip = keypoints.landmark[24]

        center_shoulder = (left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2
        center_hip = (left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2

        center = (center_shoulder[0] + center_hip[0]) / 2, (center_shoulder[1] + center_hip[1]) / 2

        O_center_shoulder = (center_shoulder[0] - center[0], center_shoulder[1] - center[1])
        Oy = (0, -1)

        theta = self.calculate_phase_difference(O_center_shoulder, Oy)

        # rotate each key point
        for point in keypoints.landmark:
            point_rotated = self.rotate_point((point.x, point.y), center, theta, origin_size)
            point.x = point_rotated[0]
            point.y = point_rotated[1]

        return keypoints
    
    def square_for_image(self, original_image):
        # Convert OpenCV image (BGR) to PIL image (RGB)
        original_image_pil = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        # Get original image size (width, height)
        original_size = original_image_pil.size

        # Find the largest dimension to make the image square
        max_size = max(original_size)

        # Create a new square image with black background
        new_image = Image.new("RGB", (max_size, max_size), (0, 0, 0))

        # Calculate the position to paste the original image (centered)
        paste_position = ((max_size - original_size[0]) // 2, (max_size - original_size[1]) // 2)

        # Paste the original image onto the new black square
        new_image.paste(original_image_pil, paste_position)

        # Convert back to OpenCV format (BGR)
        new_image_cv2 = cv2.cvtColor(np.array(new_image), cv2.COLOR_RGB2BGR)

        return new_image_cv2, (max_size, max_size)
    
    def calculate_phase_difference(self, OA: tuple, OB: tuple) -> float:
        """
        Calculate the phase difference (in degrees) of vector OB relative to vector OA.
        :param OA: A tuple representing the first vector (OA).
        :param OB: A tuple representing the second vector (OB).
        :return: Phase difference in degrees from 0 to 360.
        """
        # Calculate the dot product and magnitudes
        dot_product = OA[0] * OB[0] + OA[1] * OB[1]
        norm_OA = np.sqrt(OA[0] ** 2 + OA[1] ** 2)
        norm_OB = np.sqrt(OB[0] ** 2 + OB[1] ** 2)

        # Calculate the cosine of the angle
        cos_theta = dot_product / (norm_OA * norm_OB)
        
        # Ensure the value is within the valid range for arccos due to floating-point errors
        cos_theta = np.clip(cos_theta, -1, 1)
        
        # Calculate the angle in radians
        theta = np.arccos(cos_theta)
        
        # Calculate the cross product (only the z-component for 2D vectors)
        cross_product = OA[0] * OB[1] - OA[1] * OB[0]
        
        # Adjust the angle based on the direction
        if cross_product < 0:
            theta = 2 * np.pi - theta

        # Convert the angle to degrees
        theta_degrees = np.degrees(theta)

        return theta_degrees
    
    def rotate_point(self, point: tuple, center: tuple, angle: float, origin_size = (612, 408)) -> tuple:
        x, y = point
        cx, cy = center

        x_new = (x - cx) * np.cos(np.radians(angle)) * origin_size[0] - (y - cy) * np.sin(np.radians(angle)) * origin_size[1] + cx * origin_size[0]
        y_new = (x - cx) * np.sin(np.radians(angle)) * origin_size[0] + (y - cy) * np.cos(np.radians(angle)) * origin_size[1] + cy * origin_size[1]

        return x_new / origin_size[0], y_new / origin_size[1]