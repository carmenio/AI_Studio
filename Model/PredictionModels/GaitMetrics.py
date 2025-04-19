# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark
from typing import Dict, List, Tuple
class GaitMetrics:
    """
        Calculate gait metrics for a particular gait type.
    """
    def __init__(self, landmarks: Dict[int , List[Landmark]], gait_type: str):
        self.landmarks = landmarks 
        self.type = gait_type

    def get_landmarks(self, required_landmarks: List[int]) -> List[Landmark]:
        metric_landmarks = []

        for landmarks_list in self.landmarks.values():
            landmarks_list: List[Landmark]

            # iterate through the landmarks in the list and get the required ones using the index
            # and append them to the metric_landmarks list
            for i in range(len(landmarks_list)):
                if i in required_landmarks:
                    metric_landmarks.append(landmarks_list[i])

        return metric_landmarks
    
    def calc_metrics(self, required_landmarks: List[int], metric_eqn = None):
        metric_values = []
        
        # Parse the landmarks to get the required ones
        metric_landmarks = self.get_landmarks(required_landmarks) # 2D list of landmarks

        # Perform calculations on the metric_landmarks to get the desired metric
        # metric_value = 0.0
        for frame_req_landmarks in metric_landmarks: # gets the required landmarks list
            # frame_req_landmarks: List[Landmark]

            # Apply the metric equation if provided, otherwise default to 0.0
            metric_value = metric_eqn(frame_req_landmarks) if metric_eqn else 0.0

            metric_values.append(metric_value)

        return metric_values

    # TODO: Change the func names to gait metric name and add the equations for each metric
    
    # ! THIS IS AN EXAMPLE OF A GAIT METRIC FUNCTION
    def stepLength(self, required_landmarks: List[int]):
        """
            Calculate the ______ metric using the required landmarks.
        """

        # Lambda function to calculate step length between landmarks 0 and 1
        eqn = lambda landmarks: abs(landmarks[0].x - landmarks[1].x)

        return self.calc_metrics(required_landmarks, eqn)

    def metric2(self, required_landmarks: List[int]):
        pass