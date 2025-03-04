# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Utils.BaseClass import BaseClass

from typing import Optional, List, Union

import numpy as np
import mediapipe as mp

# --- Memory efficient Landmarks class ---
class Landmark(BaseClass):
    __slots__ = ('x', 'y', 'z', 'visibility')

    def __init__(self, x: Union[float,int], y: Union[float,int], z: Union[float,int], visibility: Union[float,int]):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility

    def __repr__(self):
        return (f"{self.__class__.__name__}(x: {self.x:.2f}, y: {self.y:.2f}, "
                f"z: {self.z:.2f}, vis: {self.visibility:.2f})")

# --- PoseFrame class ---
class PoseFrame(BaseClass):
    def __init__(self, landmarks:Optional[List]=None) -> None:
        
        if landmarks:
            self.landmarks = landmarks
        else:
            self.landmarks = []
            
        self.current_frame = 0

    # ----------- Creation of Frames ----------- #
    def _convert_landmarks(self, pose_results):
        """Convert mediapipe landmarks into Landmarks instances using a list comprehension."""
        return [Landmark(lm.x, lm.y, lm.z, lm.visibility)
                for lm in pose_results.pose_landmarks.landmark]

    def get_landmarks(self) -> np.ndarray:
        """
        Retrieve landmarks as a NumPy array of shape (N, 3).
        This avoids the overhead of Python lists if further numeric processing is needed.
        """
        
        return self.landmarks
    
    def add_landmark(self, landmark:Landmark):
        self.landmarks.append(landmark)
    
    
    def __getitem__(self, index):
        return self.landmarks[index]  # Allow indexing