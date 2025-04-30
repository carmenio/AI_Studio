# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from Model.PredictionModels.ExtractLandmarks import ExtractLandmarks
class GaitMetrics:
    """
        Calculate gait metrics for a particular gait type.
    """
    def __init__(self, landmarks_of_gait: Dict[int , List[Landmark]], video_mapping: Dict[int, int], gait_type: str):
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
    @staticmethod
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
    @staticmethod
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

        # print(heel_height)
        return heel_height
    
    def plot_heel_height(self, foot_index: int, vid_ID: int):
        """
            Plot the heel height for each frame in the video.
        """
        # Get the heel height for each frame in the video
        heel_height = self.get_heel_height_per_vid(foot_index, vid_ID)
        
        # Plot the heel height
        import matplotlib.pyplot as plt

        plt.plot(heel_height)
        plt.title(f"Heel Height for Video {vid_ID}")
        # x limits
        plt.xlim(1, len(heel_height))
        plt.xlabel("Frame")
        # y limits
        plt.ylim(min(heel_height) - 100.0, max(heel_height) + 100.0)
        plt.ylabel("Heel Height (z)")
        plt.show()

    # def calc_step_metrics(self, foot_index: int, vid_ID: int):
    #     """
    #         Find the contact points for the foot in the video.
    #     """
    #     # Get the heel height for each frame in the video
    #     heel_height = self.get_heel_height_per_vid(foot_index, vid_ID)
    #     buffer = 0.2

    #     # Find the minimum heel height in the video
    #     min_heel_height = min(heel_height)

    #     # Find the contact points (frames where heel height is less than min_heel_height + buffer)
    #     contact_ground = []  # True or False for each frame
    #     # True if foot is on the ground, False if foot is in the air
    #     for foot_height in heel_height:
    #         if foot_height < min_heel_height + buffer:
    #             contact_ground.append(True)
    #         else:
    #             contact_ground.append(False)
        
    #     # contact_ground = [False, False, False, False, True, True, True, True, True, False, False, False, True, True, True] example
    #     # find the mid points of the Trues groups
    #     mid_points_indices = []
    #     start = None
    #     for i, contact in enumerate(contact_ground):
    #         if contact and start is None:
    #             start = i
    #         elif not contact and start is not None:
    #             mid_points_indices.append((start + i) // 2)
    #             start = None

    #     # handle a true group extending to the end
    #     if start is not None:
    #         mid_points_indices.append((start + len(contact_ground)) // 2)

    #     # output: mid_points_indices = [4, 8, 12] example
        
    #     # calculate step length and step time
    #     step_lengths = []
    #     step_times = []

    #     for i in range(len(mid_points_indices) - 1):
    #         # Calculate step length (distance between mid points)
    #         step_length = abs(heel_height[mid_points_indices[i]] - heel_height[mid_points_indices[i + 1]])
    #         step_lengths.append(step_length)

    #         # Calculate step time (difference in frame indices)
    #         step_time = (mid_points_indices[i + 1] - mid_points_indices[i]) / self.fps  # in seconds
    #         step_times.append(step_time)

    #     return step_lengths, step_times, mid_points_indices, len(heel_height)
    
    # def fill_in_step_metrics(self, step_lengths: List[float], step_times: List[float], mid_points_indices: List[int], n_frames: int):
    #     """
    #         Fill in the step metrics for the entire video.
    #     """
    #     # Create a new array of size length of video (n_frames)
    #     filled_in_step_lengths = []
    #     filled_in_step_times = []

    #     # Fill in the gaps using the step lengths and times
    #     for i in range(n_frames):
    #         # Find what index is needed for step_info_index
    #         step_info_index = next((index for index, value in enumerate(mid_points_indices) if value > i), None)
            
    #         if step_info_index is not None:
    #             filled_in_step_lengths.append(step_lengths[step_info_index])
    #             filled_in_step_times.append(step_times[step_info_index])
    #         else:
    #             filled_in_step_lengths.append(0)
    #             filled_in_step_times.append(0)

    #     return filled_in_step_lengths, filled_in_step_times
    
    #     # * EXAMPLE
    #     # step_lengths = [0.32, 0.29, 0.35]  # Lengths in meters between consecutive steps
    #     # step_times = [0.5, 0.47, 0.52]     # Times in seconds between consecutive steps
    #     # mid_points_indices = [10, 25, 40]  # Frame indices where foot contacts occur
    #     # n_frames = 50                      # Total number of frames in video

    #     # Frames 0-9
    #     # For each frame, step_info_index = 0 (because mid_points_indices[0] = 10 > current frame)
    #     # Each frame gets: step_lengths[0] = 0.32 and step_times[0] = 0.5
    #     # Frames 10-24
    #     # For each frame, step_info_index = 1 (because mid_points_indices[1] = 25 > current frame)
    #     # Each frame gets: step_lengths[1] = 0.29 and step_times[1] = 0.47
    #     # Frames 25-39
    #     # For each frame, step_info_index = 2 (because mid_points_indices[2] = 40 > current frame)
    #     # Each frame gets: step_lengths[2] = 0.35 and step_times[2] = 0.52
    #     # Frames 40-49
    #     # For each frame, step_info_index = None (no contact points after these frames)
    #     # Each frame gets: 0 for both step length and time

    #     # filled_in_step_lengths = [
    #     #     # Frames 0-9 (10 frames)
    #     #     0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32, 0.32,
    #     #     # Frames 10-24 (15 frames)
    #     #     0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29, 0.29,
    #     #     # Frames 25-39 (15 frames)
    #     #     0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35, 0.35,
    #     #     # Frames 40-49 (10 frames)
    #     #     0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    #     # ]
    
    # def calc_stride_metrics(self, foot_index: int, vid_ID: int
    #                     ) -> Tuple[List[float], List[float], List[int], int]:
    #     """
    #     Returns:
    #         stride_lengths     : horizontal distances between successive foot contacts
    #         stride_times       : times (in s) between those contacts
    #         mid_points_idx   : frame‐indices of each contact midpoint
    #         n_frames         : total frames in this video
    #     """
    #     # 1) pull out only the frames for this video
    #     frames = [f for f in self.landmarks.keys() if self.video_mapping[f] == vid_ID]
    #     frames.sort()
        
    #     # 2) collect positions per frame
    #     positions = []  # list of (x, y, z)
    #     for f in frames:
    #         lm = self.landmarks[f][foot_index]
    #         positions.append((lm.x, lm.y, lm.z))
    #     n_frames = len(positions)
        
    #     # 3) find “on‐ground” via z
    #     zs = [p[2] for p in positions]
    #     BUFFER = 0.2  # buffer to account for noise in z values
    #     thresh = min(zs) + BUFFER
    #     contact = [z < thresh for z in zs]
        
    #     # 4) extract midpoints of each True‐run
    #     mids = []
    #     start = None
    #     for i, c in enumerate(contact):
    #         if c and start is None:
    #             start = i
    #         elif not c and start is not None:
    #             mids.append((start + i) // 2)
    #             start = None
    #     if start is not None:
    #         mids.append((start + n_frames) // 2)
        
    #     # 5) compute strides between consecutive mids
    #     stride_lengths = []
    #     stride_times   = []
    #     for i in range(len(mids) - 1):
    #         x0, y0, _ = positions[mids[i]]
    #         x1, y1, _ = positions[mids[i+1]]
    #         # horizontal distance
    #         dist = np.hypot(x1 - x0, y1 - y0)
    #         stride_lengths.append(dist)
    #         # time
    #         dt = (mids[i+1] - mids[i]) / self.fps
    #         stride_times.append(dt)
        
    #     return stride_lengths, stride_times, mids, n_frames

    # def fill_in_stride_metrics(self,
    #                          stride_lengths: List[float],
    #                          stride_times:   List[float],
    #                          mids:         List[int],
    #                          n_frames:     int
    #                         ) -> Tuple[List[float], List[float]]:
    #     """
    #     For each frame, assign the upcoming (or previous) stride’s length & time.
    #     Frames ≥ last contact get zeros.
    #     """
    #     filled_L = [0.0] * n_frames
    #     filled_T = [0.0] * n_frames

    #     if not mids or not stride_lengths:
    #         return filled_L, filled_T

    #     # define interval boundaries
    #     # interval 0: frames [0 .. mids[1]-1] → step_lengths[0]
    #     # interval 1: frames [mids[1] .. mids[2]-1] → step_lengths[1], etc.
    #     # after mids[-1]: leave zeros
        
    #     for i in range(n_frames):
    #         # find which interval i belongs to
    #         # if before second midpoint, use stride_lengths[0]
    #         for k in range(len(stride_lengths)):
    #             if i < mids[k+1]:
    #                 filled_L[i] = stride_lengths[k]
    #                 filled_T[i] = stride_times[k]
    #                 break
    #         # else i ≥ mids[-1] → leave as zero

    #     return filled_L, filled_T
    # 1. Stride Length & Time

    def calc_stride_metrics(self, foot_index: int, vid_ID: int
                      ) -> Tuple[List[float], List[float], List[int], int]:
        """
        Returns:
            stride_lengths : horizontal distances (x–z plane) between successive foot contacts
            stride_times   : times (in seconds) between those contacts
            mids           : frame‐indices of each contact midpoint
            n_frames       : total frames in this video
        """
        # 1) gather all frames for this video, sorted
        vid_frames = sorted(f for f, v in self.video_mapping.items() if v == vid_ID)
        n_frames = len(vid_frames)

        # 2) collect each frame’s (x, y, z)
        pos = [(self.landmarks[f][foot_index].x,
                self.landmarks[f][foot_index].y,
                self.landmarks[f][foot_index].z)
            for f in vid_frames]

        # 3) detect “on ground” by y (vertical) being close to the minimum
        ys = [p[1] for p in pos]
        min_y, max_y = min(ys), max(ys)
        BUFFER = 0.3  # buffer to account for noise in y values
        thresh = min_y + (max_y - min_y) * BUFFER
        contact = [y <= thresh for y in ys]

        # 4) find midpoint of each True‐run
        mids = []
        start = None
        for i, c in enumerate(contact):
            if c and start is None:
                start = i
            elif not c and start is not None:
                mids.append((start + i)//2)
                start = None
        if start is not None:
            mids.append((start + len(contact))//2)

        # 5) compute stride distances & times between consecutive mids
        stride_lengths = []
        stride_times = []
        for j in range(len(mids) - 1):
            i0, i1 = mids[j], mids[j+1]
            f0, f1 = vid_frames[i0], vid_frames[i1]
            x0, _, z0 = pos[i0]
            x1, _, z1 = pos[i1]
            # horizontal (x–z) distance
            stride_lengths.append(np.hypot(x1 - x0, z1 - z0))
            # time difference
            stride_times.append((f1 - f0) / self.fps)

        return stride_lengths, stride_times, mids, n_frames


    def fill_in_stride_metrics(self,
                            stride_lengths: List[float],
                            stride_times:   List[float],
                            mids:           List[int],
                            n_frames:       int
                            ) -> Tuple[List[float], List[float]]:
        """
        For each frame index 0..n_frames-1, assign the upcoming
        stride’s length/time. After the final contact, values remain 0.0.
        """
        filled_L = [0.0] * n_frames
        filled_T = [0.0] * n_frames

        if len(mids) < 2:
            return filled_L, filled_T

        # Determine, for each frame index i, which interval k it’s in:
        #  interval k covers: mids[k] ≤ i < mids[k+1]
        # interval_idx points to which stride‐interval you’re in:

        # Interval 0 covers frames from 0 up to (but not including) 
        # the second contact midpoint.
        # Interval 1 covers frames from the 2nd through just before the 3rd midpoint, etc.

        interval_idx = 0
        for i in range(n_frames):
            # move to next interval if passed its start
            while interval_idx + 1 < len(mids) and i >= mids[interval_idx+1]:
                interval_idx += 1
            # only assign if there is a next stride available
            if interval_idx < len(stride_lengths):
                filled_L[i] = stride_lengths[interval_idx]
                filled_T[i] = stride_times[interval_idx]

        return filled_L, filled_T


    def calc_stride_metrics_for_video(self, foot_index: int, vid_ID: int):
        """
            Calculate the stride metrics for a specific video.
        """
        # Get the stride metrics for the video
        stride_lengths, stride_times, mid_points_indices, n_frames = self.calc_stride_metrics(foot_index, vid_ID)

        # Fill in the stride metrics for the entire video
        filled_in_stride_lengths, filled_in_stride_times = self.fill_in_stride_metrics(stride_lengths, stride_times, mid_points_indices, n_frames)

        return filled_in_stride_lengths, filled_in_stride_times
    
    def calc_stride_metrics_for_all_videos(self, foot_index: int):
        """
            Calculate the stride metrics for all videos.
        """
        all_stride_metrics = {
            f"strideLength": [],
            f"strideTime": []
        }

        # Iterate through each video ID and calculate the stride metrics
        for vid_ID in set(self.video_mapping.values()):
            stride_lengths, stride_times = self.calc_stride_metrics_for_video(foot_index, vid_ID)
            # Append the stride metrics to the dictionary
            all_stride_metrics[f"strideLength"].extend(stride_lengths)
            all_stride_metrics[f"strideTime"].extend(stride_times)

        return all_stride_metrics

    # -------------------------------------------------------------------------------------------------------------------- #

    def create_metrics_list(self, rightHipAngle_req_landmarks: List[int], leftHipAngle_req_landmarks: List[int],
                            rightKneeAngle_req_landmarks: List[int], leftKneeAngle_req_landmarks: List[int],
                            bodyTiltAngle_req_landmarks: List[int], headTiltAngle_req_landmarks: List[int], stride_foot_index: int = 29):
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
        metrics_dict.update(self.calc_stride_metrics_for_all_videos(stride_foot_index))

        # create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame(metrics_dict)
        
        # Add gait type as a column
        metrics_df['Gait Type'] = self.type

        # add video ID as well
        metrics_df['Video ID'] = [self.video_mapping[frame] for frame in self.landmarks.keys()]

        return metrics_df


# example
if __name__ == "__main__":
    GAIT_PATH = "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait"
    extractor = ExtractLandmarks("Normal_Gait", GAIT_PATH)

    pkl_files = sorted([os.path.join(GAIT_PATH, file) for file in os.listdir(GAIT_PATH) if file.endswith('.pkl')])

    # print(pkl_files)  # Print the list of pkl files
    # Load the pkl files into the extractor
    extractor.load_pkl_files(pkl_files)
    # Load the pose sequences from the pkl files
    extractor.extract()

    # NOTE: Now we have the landmarks for the gait type, we can calculate the metrics
    gait_norm = GaitMetrics(extractor.landmarks, extractor.video_mapping, "Normal_Gait")

    # # plot the heel height for each video
    # gait_norm.plot_heel_height(29, 2)  

    df = gait_norm.create_metrics_list(
        rightHipAngle_req_landmarks=[11, 23, 25], 
        leftHipAngle_req_landmarks=[12, 24, 26],
        rightKneeAngle_req_landmarks=[23, 25, 27], 
        leftKneeAngle_req_landmarks=[24, 26, 28],
        bodyTiltAngle_req_landmarks=[11, 12, 23, 24], 
        headTiltAngle_req_landmarks=[0, 11, 12]
    )
    
    # save the DataFrame to a csv file
    df.to_csv("D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait/Normal_Gait_Metrics.csv", index=False)

# use extractor.video_mapping to get the video ID for each frame when creating the dataset
# TODO: After calculating metrics for each gait type, we r left with GaitMetrics objects for each gait type. 
# Use these objects in the main file to create a dataset by concatenating the metrics for each gait type.