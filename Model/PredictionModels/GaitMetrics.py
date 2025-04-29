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
    def __init__(self, landmarks_of_gait: Dict[int , List[Landmark]], video_mapping: Dict[int, str], gait_type: str):
        self.landmarks = landmarks_of_gait 
        self.video_mapping = video_mapping  # Maps timestamps to source video IDs
        self.type = gait_type  # Gait type (e.g., "Antalgic_Gait", "Normal_Gait", etc.)
        self.fps = 30  # Frames per second (FPS) for a video
        
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

    def get_heel_height_per_vid(self, foot_index: int, vid_ID: int):
        """
            Get the heel height for each frame in the video.
            foot_index: 29 for right foot or 30 for left foot
        """
        # Get the heel height for each frame in the video
        heel_height = [landmark[foot_index].z for frame, landmark in self.landmarks.items() if self.video_mapping[frame] == vid_ID]

        return heel_height

    def calc_step_metrics(self, foot_index: int, vid_ID: int):
        """
            Find the contact points for the foot in the video.
            foot_index: 29 for right foot or 30 for left foot
        """
        # Get the heel height for each frame in the video
        heel_height = self.get_heel_height_per_vid(foot_index, vid_ID)
        buffer = 0.2

        # Find the minimum heel height in the video
        min_heel_height = min(heel_height)

        # Find the contact points (frames where heel height is less than min_heel_height + buffer)
        contact_ground = []  # True or False for each frame
        # True if foot is on the ground, False if foot is in the air
        for foot_height in heel_height:
            if foot_height < min_heel_height + buffer:
                contact_ground.append(True)
            else:
                contact_ground.append(False)
        
        # contact_ground = [False, False, False, False, True, True, True, True, True, False, False, False, True, True, True] example
        # find the mid points of the Trues groups
        mid_points_indices = []
        start = None
        for i, contact in enumerate(contact_ground):
            if contact and start is None:
                start = i
            elif not contact and start is not None:
                mid_points_indices.append((start + i) // 2)
                start = None

        # handle a true group extending to the end
        if start is not None:
            mid_points_indices.append((start + len(contact_ground)) // 2)

        # output: mid_points_indices = [4, 8, 12] example
        
        # calculate step length and step time
        step_lengths = []
        step_times = []

        for i in range(len(mid_points_indices) - 1):
            # Calculate step length (distance between mid points)
            step_length = abs(heel_height[mid_points_indices[i]] - heel_height[mid_points_indices[i + 1]])
            step_lengths.append(step_length)

            # Calculate step time (difference in frame indices)
            step_time = (mid_points_indices[i + 1] - mid_points_indices[i]) / self.fps  # in seconds
            step_times.append(step_time)

        return step_lengths, step_times, mid_points_indices, len(heel_height)
    
    def fill_in_step_metrics(self, step_lengths: List[float], step_times: List[float], mid_points_indices: List[int], n_frames: int):
        """
            Fill in the step metrics for the entire video.
        """
        # Create a new array of size length of video (n_frames)
        filled_in_step_lengths = []
        filled_in_step_times = []

        # Fill in the gaps using the step lengths and times
        for i in range(n_frames):
            # Find what index is needed for step_info_index
            step_info_index = next((index for index, value in enumerate(mid_points_indices) if value > i), None)
            
            if step_info_index is not None:
                filled_in_step_lengths.append(step_lengths[step_info_index])
                filled_in_step_times.append(step_times[step_info_index])
            else:
                filled_in_step_lengths.append(0)
                filled_in_step_times.append(0)

        return filled_in_step_lengths, filled_in_step_times
    
        # * EXAMPLE
        # step_lengths = [0.32, 0.29, 0.35]  # Lengths in meters between consecutive steps
        # step_times = [0.5, 0.47, 0.52]     # Times in seconds between consecutive steps
        # mid_points_indices = [10, 25, 40]  # Frame indices where foot contacts occur
        # n_frames = 50                      # Total number of frames in video

        # Frames 0-9
        # For each frame, step_info_index = 0 (because mid_points_indices[0] = 10 > current frame)
        # Each frame gets: step_lengths[0] = 0.32 and step_times[0] = 0.5
        # Frames 10-24
        # For each frame, step_info_index = 1 (because mid_points_indices[1] = 25 > current frame)
        # Each frame gets: step_lengths[1] = 0.29 and step_times[1] = 0.47
        # Frames 25-39
        # For each frame, step_info_index = 2 (because mid_points_indices[2] = 40 > current frame)
        # Each frame gets: step_lengths[2] = 0.35 and step_times[2] = 0.52
        # Frames 40-49
        # For each frame, step_info_index = None (no contact points after these frames)
        # Each frame gets: 0 for both step length and time

        # filled_in_step_lengths = [
        #     # Frames 0-9 (10 frames)
        #     0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
        #     # Frames 10-24 (15 frames)
        #     0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29,
        #     # Frames 25-39 (15 frames)
        #     0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
        #     # Frames 40-49 (10 frames)
        #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        # ]
    
    def calc_step_metrics_for_video(self, foot_index: int, vid_ID: int):
        """
            Calculate the step metrics for a specific video.
        """
        # Get the step metrics for the video
        step_lengths, step_times, mid_points_indices, n_frames = self.calc_step_metrics(foot_index, vid_ID)

        # Fill in the step metrics for the entire video
        filled_in_step_lengths, filled_in_step_times = self.fill_in_step_metrics(step_lengths, step_times, mid_points_indices, n_frames)

        return filled_in_step_lengths, filled_in_step_times
    
    def calc_step_metrics_for_all_videos(self, foot_index: int, foot_type: str):
        """
            Calculate the step metrics for all videos.
        """
        all_step_metrics = {}

        # Iterate through each video ID and calculate the step metrics
        for vid_ID in set(self.video_mapping.values()):
            step_lengths, step_times = self.calc_step_metrics_for_video(foot_index, vid_ID)
            all_step_metrics[vid_ID] = {
                f"{foot_type}StepLength": step_lengths,
                f"{foot_type}StepTime": step_times
            }

        return all_step_metrics

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
            
            # Append values to corresponding lists
            metrics_dict["RightHipAngle"].append(right_hip_angle)
            metrics_dict["LeftHipAngle"].append(left_hip_angle)
            metrics_dict["RightKneeAngle"].append(right_knee_angle)
            metrics_dict["LeftKneeAngle"].append(left_knee_angle)
            metrics_dict["BodyLeanAngle"].append(body_lean_angle)
            metrics_dict["HeadTiltAngle"].append(head_tilt_angle)

        # add the step metrics to the dictionary
        metrics_dict.update(self.calc_step_metrics_for_all_videos(29, "Right")) # Right leg
        metrics_dict.update(self.calc_step_metrics_for_all_videos(30, "Left")) # Left leg

        # create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame(metrics_dict)
        
        # Add gait type as a column
        metrics_df['Gait Type'] = self.type

        # add video ID as well
        metrics_df['Video ID'] = [self.video_mapping[frame] for frame in self.landmarks.keys()]

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

# use extractor.video_mapping to get the video ID for each frame when creating the dataset
# TODO: After calculating metrics for each gait type, we r left with GaitMetrics objects for each gait type. 
# Use these objects in the main file to create a dataset by concatenating the metrics for each gait type.