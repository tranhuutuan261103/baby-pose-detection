import pandas as pd
import numpy as np
import mediapipe as mp
mp_pose = mp.solutions.pose

class PoseScalerHelper:
    def __init__(self, IMPORTANT_LMS):
        self.IMPORTANT_LMS = IMPORTANT_LMS

    def extract_and_recalculate_landmarks(self, pose_landmarks):
        # Prepare data for CSV
        columns_name = [] + [f"{landmark}_{axis}" for landmark in self.IMPORTANT_LMS for axis in ['x', 'y', 'z']]
        columns_value = []

        for id, landmark in enumerate(pose_landmarks):
            if mp_pose.PoseLandmark(id).name.lower() in self.IMPORTANT_LMS:
                # Adjust keypoint coordinates
                columns_value.extend([landmark.x, landmark.y, landmark.z])

        df_key_points = pd.DataFrame([columns_value], columns=columns_name)

        center = self.find_center_of_mass(df_key_points)
        shifting = (0.5 - center[0], 0.5 - center[1])

        for landmark in self.IMPORTANT_LMS:
            df_key_points[f"{landmark}_x"] += shifting[0]
            df_key_points[f"{landmark}_y"] += shifting[1]

        # Calculate the scale value based on the distance between nose and hip
        scale_value = self.calculate_scale_value(df_key_points)

        # Scale keypoints
        df_key_points = self.scale_keypoints(df_key_points, scale_value)

        return df_key_points
    
    def find_center_of_mass(self, df_key_points):
        left_hip = (df_key_points["left_hip_x"], df_key_points["left_hip_y"])
        right_hip = (df_key_points["right_hip_x"], df_key_points["right_hip_y"])
        
        # Find the center of the hips
        center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Get the nose coordinates
        nose = (df_key_points["nose_x"], df_key_points["nose_y"])

        return (center_hip[0] + nose[0]) / 2, (center_hip[1] + nose[1]) / 2

    def calculate_scale_value(self, df_key_points):
        """Calculate the scale value based on nose and hip points."""
        left_hip = (df_key_points["left_hip_x"], df_key_points["left_hip_y"])
        right_hip = (df_key_points["right_hip_x"], df_key_points["right_hip_y"])
        
        # Find the center of the hips
        center_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        # Get the nose coordinates
        nose = (df_key_points["nose_x"], df_key_points["nose_y"])

        # Calculate the distance between nose and center of the hips
        distance = np.sqrt((center_hip[0] - nose[0])**2 + (center_hip[1] - nose[1])**2)

        # Scale value based on distance
        return 0.5 / distance

    def scale_keypoints(self, df_key_points, scale_value):
        """Scale all key points based on the calculated scale value."""
        for landmark in self.IMPORTANT_LMS:
            df_key_points[f"{landmark}_x"] *= scale_value
            df_key_points[f"{landmark}_y"] *= scale_value
        return df_key_points