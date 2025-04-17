# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import Landmark, PoseFrame
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

    # TODO: Change the func names to gait metric name and add the equations for each metric
    
    def metric1(self, required_landmarks: List[int]):
        """
            Calculate the ______ metric using the required landmarks.
        """
        # Parse the landmarks to get the required ones
        metric_landmarks = self.get_landmarks(required_landmarks)

        # Perform calculations on the metric_landmarks to get the desired metric
        metric_value = 0.0

        # equation here


        return metric_value

    def metric2(self, required_landmarks: List[int]):
        pass