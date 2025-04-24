# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark
from typing import Dict, List
import numpy as np
class GaitMetrics:
    """
        Calculate gait metrics for a particular gait type.
    """
    def __init__(self, landmarks_of_gait: Dict[int , List[Landmark]]):
        self.landmarks = landmarks_of_gait 
        self.metrics_list = [] # a list to store the metrics for each frame

    # NOTE: I Like it, but the issue is it too slow and you cant control the order of the landmarks, important when you are cal angles
    def get_landmarks(self, landmarks_list: List[Landmark], required_landmarks: List[int]) -> List[Landmark]:
        """
            Get the required landmarks from the landmarks list.
        """
        metric_landmarks = []

        # iterate through the landmarks in the list and get the required ones using the index
        # and append them to the metric_landmarks list
        for i in required_landmarks:
            # if i in required_landmarks:
            metric_landmarks.append(landmarks_list[i])

        return metric_landmarks
    
    
    def calc_metrics(self, landmarks_list: List[Landmark], required_landmarks: List[int], metric_eqn: callable = None) -> float:
        """
            Calculate the gait metric using the required landmarks and the metric equation.
            If no equation is provided, return 0.0.
        """
        metric_value = 0.0
        
        # Parse the landmarks to get the required ones
        metric_landmarks = self.get_landmarks(landmarks_list, required_landmarks) # 1D list of landmarks

        # Perform calculations on the metric_landmarks to get the desired metric value
        # Apply the metric equation if provided, otherwise default to 0.0
        metric_value = metric_eqn(metric_landmarks) 

        return metric_value

    # Kinematic Metrics
    # helper function to calculate the angle between three points in 3D space
    def calculate_angle_3d(point1, point2, point3):
        """
        Calculate the angle between three 3D points where point2 is the vertex.

        Args:
            point1, point2, point3: 3D points as [x, y, z]

        Returns:
            Angle in degrees
        """
        
        # Create vectors from points
        vector1 = np.array([point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]])
        vector2 = np.array([point3[0] - point2[0], point3[1] - point2[1], point3[2] - point2[2]])
        
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(vector1)
        magnitude2 = np.linalg.norm(vector2)
        
        # Calculate angle in radians and convert to degrees
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle)

    # NOTE: The order of the landmarks is important for the angle calculation

    # 1. Hig Angle - Need shoulder, hip, knee landmarks
    def hipAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the hip angle metric using the required landmarks.
        """

        # Lambda function to calculate hip angle
        eqn = lambda landmarks: self.calculate_angle_3d(
            [landmarks[required_landmarks[0]].x, landmarks[required_landmarks[0]].y, landmarks[required_landmarks[0]].z],  # Shoulder
            [landmarks[required_landmarks[1]].x, landmarks[required_landmarks[1]].y, landmarks[required_landmarks[1]].z],  # Hip
            [landmarks[required_landmarks[2]].x, landmarks[required_landmarks[2]].y, landmarks[required_landmarks[2]].z]   # Knee
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)
    
    # 2. Knee Angle - need hip, knee, ankle landmarks
    def kneeAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the knee angle metric using the required landmarks.
        """

        # Lambda function to calculate knee angle
        eqn = lambda landmarks: self.calculate_angle_3d(
            [landmarks[required_landmarks[0]].x, landmarks[required_landmarks[0]].y, landmarks[required_landmarks[0]].z],  # Hip
            [landmarks[required_landmarks[1]].x, landmarks[required_landmarks[1]].y, landmarks[required_landmarks[1]].z],  # Knee
            [landmarks[required_landmarks[2]].x, landmarks[required_landmarks[2]].y, landmarks[required_landmarks[2]].z]   # Ankle
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)
    
    

    # def stepLength(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
    #     """
    #         Calculate the ______ metric using the required landmarks.
    #     """

    #     # Lambda function to calculate step length between landmarks 0 and 1
    #     eqn = lambda landmarks: abs(landmarks[0].x - landmarks[1].x)

    #     return self.calc_metrics(landmarks_list, required_landmarks, eqn)

    # def metric2(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
    #     pass

    # # .... other metrics

    # ! CHANGE THE ARGS TO PROPER GAIT METRIC NAMES & ADD ARGS FOR EACH METRIC
    def create_metrics_list(self, metric1_req_landmarks: List[int], metric2_req_landmarks: List[int], ..., ...):
        """
            Create a list of metrics for the gait type.
        """
        
        # for each frame, calc the metrics using the required landmarks and append them to the metrics_list
        for frame_landmarks_list in self.landmarks.values():
            frame_landmarks_list : List[Landmark]

            # iterate through the landmarks in the list and get the required ones using the index
            # and append them to the metric_landmarks list
            metric_values_for_frame = []

            metric_values_for_frame.append(self.stepLength(metric1_req_landmarks, frame_landmarks_list))
            metric_values_for_frame.append(self.metric2(metric2_req_landmarks, frame_landmarks_list))
            ...
            ...
            ...

            # >>... other metrics

            # after calculating the metrics for each frame, append them to the metrics_list
            self.metrics_list.append(metric_values_for_frame)

        # NOTE: Need to convert to a df at the end of the for loop
        


# example
if __name__ ==  'main':
    GAIT_PATH = "Model/PredictionModels/Sequences/Apr_10_Antalgic_Gait"
    extractor = ExtractLandmarks("Antalgic_Gait", GAIT_PATH)

    pkl_files = [sorted(os.path.join(GAIT_PATH, file) for file in os.listdir(GAIT_PATH) if file.endswith('.pkl'))]

    # Load the pkl files into the extractor
    extractor.load_pkl_files(pkl_files)
    # Load the pose sequences from the pkl files
    extractor.extract()

    # NOTE: Now we have the landmarks for the gait type, we can calculate the metrics
    gait_metrics_for_antalgic_gait = GaitMetrics(extractor.landmarks)
    gait_metrics_for_antalgic_gait.create_metrics_list(metric1landmarks, metrics2landmarks, ....)

# TODO: After calculating metrics for each gait type, we r left with GaitMetrics objects for each gait type. 
# Use these objects in the main file to create a dataset by concatenating the metrics for each gait type.