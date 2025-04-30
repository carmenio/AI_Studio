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

    def find_optimal_buffer(self, foot_index: int, vid_ID: int):
        """Find the optimal buffer value through visual inspection"""
        import matplotlib.pyplot as plt
        
        # Get position data
        frames = sorted(f for f, v in self.video_mapping.items() if v == vid_ID)
        pos = [(self.landmarks[f][foot_index].x,
                self.landmarks[f][foot_index].y,
                self.landmarks[f][foot_index].z) for f in frames]
        
        # Extract y values
        ys = [p[1] for p in pos]
        min_y, max_y = min(ys), max(ys)
        
        # Test different buffer values
        buffer_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7]
        fig, axs = plt.subplots(len(buffer_values), 1, figsize=(5, 10), sharex=True)

        plt.subplots_adjust(hspace=0.5)
        for i, buffer in enumerate(buffer_values):
            thresh = min_y + (max_y - min_y) * buffer
            contact = [y <= thresh for y in ys]
            
            # Plot y trajectory
            axs[i].plot(ys, label='Foot Height')
            
            # Plot threshold line
            axs[i].axhline(y=thresh, color='r', linestyle='--', label=f'Threshold (Buffer={buffer})')
            
            # Mark contact points
            contact_indices = [j for j, c in enumerate(contact) if c]
            axs[i].scatter(contact_indices, [ys[j] for j in contact_indices], 
                        color='g', marker='o', label='Contact')
            
            axs[i].set_title(f'Buffer = {buffer}')
            axs[i].legend()
        
        plt.tight_layout()
        plt.show()
        
        # get best buffer value from the user
        best_buffer = float(input("Enter the best buffer value: "))
        print(f"Best buffer value: {best_buffer}")
        self.best_buffer = best_buffer
    
    # 1. Stride Metrics

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
        BUFFER = self.best_buffer  # buffer to account for noise in y values
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
            "strideLength": [],
            "strideTime": []
        }

        # Iterate through each video ID and calculate the stride metrics
        for vid_ID in set(self.video_mapping.values()):
            stride_lengths, stride_times = self.calc_stride_metrics_for_video(foot_index, vid_ID)
            # Append the stride metrics to the dictionary
            all_stride_metrics[f"strideLength"].extend(stride_lengths)
            all_stride_metrics[f"strideTime"].extend(stride_times)

        return all_stride_metrics

    # 2. Step Metrics (right→left) 
    def calc_step_metrics_both(self,
                               right_index: int,
                               left_index:  int,
                               vid_ID:      int
                              ) -> Tuple[List[float], List[float], List[int]]:
        """
        Interleaves right→left contacts to compute true step lengths & times.
        Returns:
          step_lengths, step_times, contact_frames
        """
        # get each foot’s contact mids (uses existing stride logic)
        _, _, mids_r, _ = self.calc_stride_metrics(right_index, vid_ID)
        _, _, mids_l, _ = self.calc_stride_metrics(left_index,  vid_ID)

        # map frames→(x,z) for each foot
        frames = sorted(f for f, v in self.video_mapping.items() if v == vid_ID)
        posR   = {f: (self.landmarks[f][right_index].x, self.landmarks[f][right_index].z) for f in frames}
        posL   = {f: (self.landmarks[f][left_index].x,  self.landmarks[f][left_index].z)  for f in frames}

        # merge and sort all contacts
        contacts = []
        for i in mids_r:
            f = frames[i]
            contacts.append((f, *posR[f]))
        for i in mids_l:
            f = frames[i]
            contacts.append((f, *posL[f]))
        contacts.sort(key=lambda x: x[0])

        # compute step lengths & times between successive contacts
        step_lengths, step_times = [], []
        for k in range(len(contacts)-1):
            f0, x0, z0 = contacts[k]
            f1, x1, z1 = contacts[k+1]
            step_lengths.append(np.hypot(x1-x0, z1-z0))
            step_times.append((f1 - f0) / self.fps)

        contact_frames = [c[0] for c in contacts]
        return step_lengths, step_times, contact_frames

    def fill_in_step_metrics(self,
                    step_lengths: List[float],
                    step_times: List[float],
                    contact_frames: List[int],
                    n_frames: int,
                    vid_ID: int) -> Tuple[List[float], List[float]]:
        """
        For each frame index 0..n_frames-1, assign the upcoming
        step's length/time. After the final contact, values remain 0.0.
        """
        filled_L = [0.0] * n_frames
        filled_T = [0.0] * n_frames

        if len(contact_frames) < 2:
            return filled_L, filled_T

        # Convert contact_frames to indices in the frames list
        frames = sorted(f for f, v in self.video_mapping.items() if v == vid_ID)
        contact_indices = [frames.index(f) if f in frames else 0 for f in contact_frames]

        interval_idx = 0
        for i in range(n_frames):
            # move to next interval if passed its start
            while interval_idx + 1 < len(contact_indices) and i >= contact_indices[interval_idx+1]:
                interval_idx += 1
            # only assign if there is a next step available
            if interval_idx < len(step_lengths):
                filled_L[i] = step_lengths[interval_idx]
                filled_T[i] = step_times[interval_idx]

        return filled_L, filled_T
    
    def calc_step_metrics_for_video(self, right_index: int, left_index: int, vid_ID: int):
        """
        Calculate the step metrics for a specific video.
        
        Args:
            right_index: Index of the right foot landmark (typically 29)
            left_index: Index of the left foot landmark (typically 30)
            vid_ID: ID of the video to analyze
        
        Returns:
            filled_step_lengths: List of step lengths for each frame
            filled_step_times: List of step times for each frame
        """
        # Get all frames for this video
        frames = sorted(f for f, v in self.video_mapping.items() if v == vid_ID)
        n_frames = len(frames)
        
        # Get the step metrics for the video
        step_lengths, step_times, contact_frames = self.calc_step_metrics_both(right_index, left_index, vid_ID)
        
        # Fill in the step metrics for the entire video
        filled_step_lengths, filled_step_times = self.fill_in_step_metrics(
            step_lengths, step_times, contact_frames, n_frames, vid_ID
        )
        
        return filled_step_lengths, filled_step_times
    
    def calc_step_metrics_for_all_videos(self, right_index: int, left_index: int):
        """
        Calculate the step metrics for all videos.
        
        Args:
            right_index: Index of the right foot landmark (typically 29)
            left_index: Index of the left foot landmark (typically 30)
        
        Returns:
            Dictionary containing step lengths and step times for all frames
        """
        all_step_metrics = {
            "stepLength": [],
            "stepTime": []
        }
        
        # Iterate through each video ID and calculate the step metrics
        for vid_ID in set(self.video_mapping.values()):
            step_lengths, step_times = self.calc_step_metrics_for_video(right_index, left_index, vid_ID)
            
            # Append the step metrics to the dictionary
            all_step_metrics["stepLength"].extend(step_lengths)
            all_step_metrics["stepTime"].extend(step_times)
        
        return all_step_metrics

    # 3. Cadence
    def calc_cadence(self, step_times: List[float]) -> float:
        """ steps per minute = (number_of_steps / total_time_s) * 60 """
        if not step_times:
            return 0.0
        total = sum(step_times)
        return len(step_times) / total * 60.0

    # 4. Gait Speed
    def calc_gait_speed(self,
                        step_lengths: List[float],
                        step_times:   List[float]
                       ) -> float:
        """ speed (m/s) = sum(step_lengths) / total_time_s """
        if not step_times:
            return 0.0
        total = sum(step_times)
        return sum(step_lengths) / total
    
    def calc_cadence_gait_speed_for_video(self, right_index: int, left_index: int, vid_ID: int) -> Tuple[float, float]:
        """
        Calculate cadence and gait speed for a specific video.
        
        Args:
            right_index: Index of the right foot landmark (typically 29)
            left_index: Index of the left foot landmark (typically 30)
            vid_ID: ID of the video to analyze
        
        Returns:
            cadence: Steps per minute
            gait_speed: Walking speed in m/s
        """
        # Get the step metrics for the video (raw step data, not filled in)
        step_lengths, step_times, _ = self.calc_step_metrics_both(right_index, left_index, vid_ID)
        
        # Calculate cadence and gait speed
        cadence = self.calc_cadence(step_times)
        gait_speed = self.calc_gait_speed(step_lengths, step_times)
        
        return cadence, gait_speed
    
    def calc_cadence_gait_speed_for_all_videos(self, right_index: int, left_index: int) -> Dict[str, Dict[int, float]]:
        """
        Calculate cadence and gait speed for all videos.
        
        Args:
            right_index: Index of the right foot landmark (typically 29)
            left_index: Index of the left foot landmark (typically 30)
        
        Returns:
            Dictionary mapping video IDs to their cadence and gait speed values
        """
        cadence_by_video = {}
        gait_speed_by_video = {}
        
        # Iterate through each video ID and calculate the metrics
        for vid_ID in set(self.video_mapping.values()):
            cadence, gait_speed = self.calc_cadence_gait_speed_for_video(right_index, left_index, vid_ID)
            
            # Store the metrics by video ID
            cadence_by_video[vid_ID] = cadence
            gait_speed_by_video[vid_ID] = gait_speed
        
        return {
            "Cadence": cadence_by_video,
            "GaitSpeed": gait_speed_by_video
        }
        
    # -------------------------------------------------------------------------------------------------------------------- #

    def create_metrics_df(self, rightHipAngle_req_landmarks: List[int], leftHipAngle_req_landmarks: List[int],
                            rightKneeAngle_req_landmarks: List[int], leftKneeAngle_req_landmarks: List[int],
                            bodyTiltAngle_req_landmarks: List[int], headTiltAngle_req_landmarks: List[int], foot_idx: int = 29):
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

        # add the stride metrics to the dictionary
        metrics_dict.update(self.calc_stride_metrics_for_all_videos(foot_idx))

        # add the step metrics to the dictionary
        metrics_dict.update(self.calc_step_metrics_for_all_videos(foot_idx+1, foot_idx))

        # create a DataFrame from the metrics dictionary
        metrics_df = pd.DataFrame(metrics_dict)

        # Add cadence and gait speed metrics (they are per-video, not per-frame)
        cadence_gait_speed = self.calc_cadence_gait_speed_for_all_videos(foot_idx+1, foot_idx)
        
        # add the video ID to the DataFrame
        metrics_df['VideoID'] = [self.video_mapping[frame] for frame in self.landmarks.keys()]

        # Map the per-video metrics to each frame based on its video ID
        metrics_df['Cadence'] = metrics_df['VideoID'].map(cadence_gait_speed['Cadence'])
        metrics_df['GaitSpeed'] = metrics_df['VideoID'].map(cadence_gait_speed['GaitSpeed'])

        # add the gait type to the DataFrame
        metrics_df['GaitType'] = self.type

        # put the video ID as the first column
        cols = metrics_df.columns.tolist()
        cols.insert(0, cols.pop(cols.index('VideoID')))
        metrics_df = metrics_df[cols]

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

    # * Inspect for each video of each gait type before entering the buffer value
    gait_norm.find_optimal_buffer(29, 3)

    # df = gait_norm.create_metrics_df(
    #     rightHipAngle_req_landmarks=[11, 23, 25], 
    #     leftHipAngle_req_landmarks=[12, 24, 26],
    #     rightKneeAngle_req_landmarks=[23, 25, 27], 
    #     leftKneeAngle_req_landmarks=[24, 26, 28],
    #     bodyTiltAngle_req_landmarks=[11, 12, 23, 24], 
    #     headTiltAngle_req_landmarks=[0, 11, 12]
    # )
    
    # # save the DataFrame to a csv file
    # df.to_csv("D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait/Normal_Gait_Metrics.csv", index=False)

# use extractor.video_mapping to get the video ID for each frame when creating the dataset
# TODO: After calculating metrics for each gait type, we r left with GaitMetrics objects for each gait type. 
# Use these objects in the main file to create a dataset by concatenating the metrics for each gait type.