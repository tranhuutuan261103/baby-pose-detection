import numpy as np

class PoseRotationHelper:
    def __init__(self):
        pass

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