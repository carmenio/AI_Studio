# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark
from typing import Dict, List
class GaitMetrics:
    """
        Calculate gait metrics for a particular gait type.
    """
    def __init__(self, landmarks: Dict[int , List[Landmark]], gait_type: str):
        self.landmarks = landmarks 
        self.type = gait_type

        self.metrics_list = [] # a list to store the metrics for each frame

    # NOTE: I Like it, but the issue is it too slow and you cant control the order of the landmarks, important when you are cal angles
    def get_landmarks(self, landmarks_list: List[Landmark], required_landmarks: List[int]) -> List[Landmark]:
        """
            Get the required landmarks from the landmarks list.
        """
        metric_landmarks = []

        # iterate through the landmarks in the list and get the required ones using the index
        # and append them to the metric_landmarks list
        for i in range(len(landmarks_list)):
            if i in required_landmarks:
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

    # TODO: Change the func names to gait metric name and add the equations for each metric
    
    # ! THIS IS AN EXAMPLE OF A GAIT METRIC FUNCTION
    def stepLength(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        """
            Calculate the ______ metric using the required landmarks.
        """

        # Lambda function to calculate step length between landmarks 0 and 1
        eqn = lambda landmarks: abs(landmarks[0].x - landmarks[1].x)

        return self.calc_metrics(landmarks_list, required_landmarks, eqn)

    def metric2(self, required_landmarks: List[int], landmarks_list: List[Landmark]):
        pass

    # .... other metrics

    # ! CHANGE THE ARGS TO PROPER GAIT METRIC NAMES & ADD ARGS FOR EACH METRIC
    def create_metrics_list(self, metric1_req_landmarks: List[int], metric2_req_landmarks: List[int]):
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

            # >>... other metrics

            # after calculating the metrics for each frame, append them to the metrics_list
            self.metrics_list.append(metric_values_for_frame)

        # NOTE: Need to convert to a df at the end of the for loop

# TODO: After calculating metrics for each gait type, we r left with GaitMetrics objects for each gait type. 
# Use these objects in the main file to create a dataset by concatenating the metrics for each gait type.