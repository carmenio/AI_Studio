# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

import numpy as np
import pickle

from Model.Utils.BaseClass import BaseClass
from Model.Utils.Checks import Optional
from Model.Utils.SpeedDecorators import monitor_speed
from Model.Utils.MediaSteams import Video, MediaStream

from Model.Pose.Sequence.PoseFrame import PoseFrame

class PoseSequence(BaseClass):
    
    def __init__(self, media_stream:Optional[MediaStream]=None) -> None:
        self.media_stream = media_stream
        
        # self.pose_detection = pose_detection
        self.poses = []
        self.current_frame = 0
        
    # ----------------------------- Getters and Setters ----------------------------- 
    def get_sequence(self):
        return self.poses
        
    def add_pose_frame(self, frame: Optional[PoseFrame]): 
        self.poses.append(frame)
    
    def save(self, file_path: str):
        """Saves the PoseSequence object to disk."""
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        print(f"PoseSequence saved to {file_path}")
    
    @staticmethod
    def load(file_path: str):
        """Loads a PoseSequence object from disk."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
        
    def __iter__(self):
        """Returns the iterator object (itself)."""
        self.current_frame = 0  # Reset iteration when a new iteration begins
        return self

    def __next__(self):
        """Returns the next pose frame or raises StopIteration."""
        if self.current_frame >= len(self.poses):  # Fix len() call
            raise StopIteration  # Stop iteration when reaching the end
        
        value = self.poses[self.current_frame]
        self.current_frame += 1
        return value
    
    
    def __getitem__(self, index):
        return self.poses[index]  # Allow indexing
    
    def __len__(self):
        return len(self.poses)
    
        
if __name__ == "__main__":
    # pose = PoseSequence()
    # pose.add_pose_frame(None)
    # pose.save("pose_sequence.pkl")
    
    loaded_pose = PoseSequence.load("pose_sequence.pkl")

    # loaded_pose.
    # print(loaded_pose)
    
    for frame in loaded_pose:
        print(frame[0])