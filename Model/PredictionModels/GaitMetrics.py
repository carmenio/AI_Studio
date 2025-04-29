# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark
from typing import Dict, List
import numpy as np
import pandas as pd
class GaitMetrics:
    """
        Calculate gait metrics for a particular gait type.
    """
    def __init__(self, landmarks_of_gait: Dict[int , List[Landmark]]):
        self.landmarks = landmarks_of_gait 
        self.metrics_list = [] # a list to store the metrics for each frame

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
    
    # * The order of the landmarks is important for the angle calculation !!

    # A) Kinematic Metrics
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

        if magnitude1 * magnitude2 == 0:
            return 0  # Avoid division by zero
        
        # Calculate angle in radians and convert to degrees
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle)

    # 1. Hig Angle
    # * Need shoulder, hip, knee landmarks
    # * To calculate right hip angle, use landmarks 11, 23, 25
    # * To calculate left hip angle, use landmarks 12, 24, 26
    def hipAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the hip angle metric using the required landmarks.
        """

        # Lambda function to calculate hip angle
        eqn = lambda landmarks: self.calculate_angle_3d(
            [landmarks[0].x, landmarks[0].y, landmarks[0].z],  # Shoulder
            [landmarks[1].x, landmarks[1].y, landmarks[1].z],  # Hip
            [landmarks[2].x, landmarks[2].y, landmarks[2].z]   # Knee
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)
    
    # 2. Knee Angle 
    # * Need hip, knee, ankle landmarks
    # * To calculate right knee angle, use landmarks 23, 25, 27
    # * To calculate left knee angle, use landmarks 24, 26, 28
    def kneeAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the knee angle metric using the required landmarks.
        """

        # Lambda function to calculate knee angle
        eqn = lambda landmarks: self.calculate_angle_3d(
            [landmarks[0].x, landmarks[0].y, landmarks[0].z],  # Hip
            [landmarks[1].x, landmarks[1].y, landmarks[1].z],  # Knee
            [landmarks[2].x, landmarks[2].y, landmarks[2].z]   # Ankle
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)
    
    # -------------------------------------------------------------------------------------------------------------------- #
    
    # B) Postural feature Metrics
    # helper function to calculate the angle in the sagittal plane
    def calculate_sagittal_angle(top_point, bottom_point, is_forward_backward=True):
        """
        Calculate the angle in the sagittal plane (forward/backward tilt).
        
        Args:
            top_point: 3D point at the top of the line [x, y, z]
            bottom_point: 3D point at the bottom of the line [x, y, z]
            is_forward_backward: If True, measures forward/backward angle; if False, measures side-to-side angle
        
        Returns:
            Angle in degrees relative to vertical
        """
        
        # For forward/backward angle, we look at the y-z plane (sagittal plane)
        # For side-to-side angle, we would look at the x-z plane (frontal plane)
        
        if is_forward_backward:
            # Project points onto the y-z plane (forward/backward)
            actual_vector = np.array([0, top_point[1] - bottom_point[1], top_point[2] - bottom_point[2]])
            vertical_vector = np.array([0, 0, -1])  # Vertical reference vector pointing down
        else:
            # Project points onto the x-z plane (side-to-side)
            actual_vector = np.array([top_point[0] - bottom_point[0], 0, top_point[2] - bottom_point[2]])
            vertical_vector = np.array([0, 0, -1])  # Vertical reference vector pointing down
        
        # Calculate dot product
        dot_product = np.dot(actual_vector, vertical_vector)
        
        # Calculate magnitudes
        magnitude1 = np.linalg.norm(actual_vector)
        magnitude2 = np.linalg.norm(vertical_vector)
        
        # Calculate angle in radians and convert to degrees
        if magnitude1 * magnitude2 == 0:
            return 0  # Avoid division by zero
            
        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        return np.degrees(angle)

    # 1. Head Tilt Angle (fwd/bwd) 
    # * Need nose (0) and neck point (midpoint between shoulders 11,12) landmarks
    def headTiltAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the forward/backward head tilt angle metric using the required landmarks.
            Measures whether the head is leaning forward or downward while walking.
        """
        # Lambda function to calculate head tilt angle in the sagittal plane
        eqn = lambda landmarks: self.calculate_sagittal_angle(
            [landmarks[0].x, landmarks[0].y, landmarks[0].z],  # Nose (0)
            # Neck point (midpoint between shoulders 11,12)
            [(landmarks[1].x + landmarks[2].x)/2, 
             (landmarks[1].y + landmarks[2].y)/2, 
             (landmarks[1].z + landmarks[2].z)/2],
            True  # True for measuring forward/backward tilt
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)

    # 2. Body Lean Angle (fwd/bwd) 
    # * Need shoulder midpoint (11,12) and hip midpoint (23,24) landmarks
    def bodyLeanAngle(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the forward/backward body lean angle metric using the required landmarks.
            Measures the degree to which the torso leans forward or backward.
        """
        # Lambda function to calculate body lean angle in the sagittal plane
        eqn = lambda landmarks: self.calculate_sagittal_angle(
            # midpoint between left and right shoulders (11, 12)
            [(landmarks[0].x + landmarks[1].x)/2, 
             (landmarks[0].y + landmarks[1].y)/2, 
             (landmarks[0].z + landmarks[1].z)/2],
            # midpoint between left and right hips (23, 24)
            [(landmarks[2].x + landmarks[3].x)/2, 
             (landmarks[2].y + landmarks[3].y)/2, 
             (landmarks[2].z + landmarks[3].z)/2],
            True  # True for measuring forward/backward lean
        )

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)
    
    # -------------------------------------------------------------------------------------------------------------------- #

    # C) Spatiotemporal Metrics
    def find_min_step_height(self, foot_index: int, landmarks_list: List[Landmark]):
        
        buffer = 0.2
        array_all_feet_index = [landmark[foot_index].z for landmark in landmarks_list]
        
        min_feet = min(array_all_feet_index)
        
        # min_feet = 0.23
        
        contact_ground = []
        
        for foot_height in array_all_feet_index:
            if foot_height < min_feet + buffer:
                contact_ground.append(True)
            else:
                contact_ground.append(False)
                
                
        # contact_ground = [False,False,False,False, True,True,True,True,True, False,False,False, ]
      
        # Find Mid Points of the Trues
        # Mid_Points = [12, 24, 65, 100]
    
        # use that to calculate step length, calculate step time 
        # step_len = [10, 10 ,10, 12, 30]
        # len_of_step = [10, 10 ,10, 12, 30]
        
        
        # fill in the gaps
        # create a new array of size length of dataframe (length of video)
        filled_in_step_len = []
        for i in range(start_frame_index, end_frame_index):
            # Find what index is needed for step_info_index
            filled_in_step_len.append(
                step_len[step_info_index]
            )
        
        
        



    # -------------------------------------------------------------------------------------------------------------------- #

    def create_metrics_list(self, rightHipAngle_req_landmarks: List[int], leftHipAngle_req_landmarks: List[int],
                            rightKneeAngle_req_landmarks: List[int], leftKneeAngle_req_landmarks: List[int],
                            bodyTiltAngle_req_landmarks: List[int], headTiltAngle_req_landmarks: List[int]):
        """
            Create a list of metrics for the gait type and convert to DataFrame.
        """

        # Initialise a dict to store metrics
        # * Add the other metrics as well *
        metrics_dict = {
            "RightHipAngle": [],
            "LeftHipAngle": [],
            "RightKneeAngle": [],
            "LeftKneeAngle": [],
            "BodyLeanAngle": [],
            "HeadTiltAngle": []
        }
        
        # for each frame, calc the metrics using the required landmarks and append them to the metrics_list
        for frame_landmarks_list in self.landmarks.values():
            frame_landmarks_list : List[Landmark]

            # For each frame, calculate the metrics and store in the dictionary
            right_hip_angle = self.hipAngle(rightHipAngle_req_landmarks, frame_landmarks_list)
            left_hip_angle = self.hipAngle(leftHipAngle_req_landmarks, frame_landmarks_list)
            right_knee_angle = self.kneeAngle(rightKneeAngle_req_landmarks, frame_landmarks_list)
            left_knee_angle = self.kneeAngle(leftKneeAngle_req_landmarks, frame_landmarks_list)
            body_lean_angle = self.bodyLeanAngle(bodyTiltAngle_req_landmarks, frame_landmarks_list)
            head_tilt_angle = self.headTiltAngle(headTiltAngle_req_landmarks, frame_landmarks_list)
            ...
            ...
            
            # Append values to corresponding lists
            metrics_dict["RightHipAngle"].append(right_hip_angle)
            metrics_dict["LeftHipAngle"].append(left_hip_angle)
            metrics_dict["RightKneeAngle"].append(right_knee_angle)
            metrics_dict["LeftKneeAngle"].append(left_knee_angle)
            metrics_dict["BodyLeanAngle"].append(body_lean_angle)
            metrics_dict["HeadTiltAngle"].append(head_tilt_angle)
            ...
            ...

        # create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame(metrics_dict)
        
        # Add gait type as a column
        metrics_df['Gait Type'] = self.type

        return metrics_df

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