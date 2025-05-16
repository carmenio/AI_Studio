
# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Utils.BaseClass import BaseClass, abstractmethod, Union
from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark
from Model.Pose.Sequence.PoseSequence import PoseSequence
import numpy as np
import math
from itertools import zip_longest 


class TrimPoseSequence(BaseClass):
    def __init__(self, pose_sequence: PoseSequence):
        self.pose_sequence = pose_sequence

    def trim(self, start_trim: int = 0, end_trim: int = 0) -> PoseSequence:
        """
        Trim the PoseSequence from the start and/or end.

        :param start_trim: Number of frames to remove from the start.
        :param end_trim: Number of frames to remove from the end.
        :return: A new trimmed PoseSequence.
        """
        total_frames = len(self.pose_sequence)

        # Ensure trimming does not exceed sequence length
        start_trim = min(start_trim, total_frames)
        end_trim = min(end_trim, total_frames - start_trim)

        # Extract the trimmed frames
        trimmed_poses = self.pose_sequence.poses[start_trim: total_frames - end_trim]

        # Create a new PoseSequence with the trimmed frames
        new_sequence = PoseSequence()
        new_sequence.poses = trimmed_poses

        return new_sequence

# Example Usage:
if __name__ == "__main__":
    original_sequence = PoseSequence()
    
    # Add some dummy pose frames
    for i in range(10):
        original_sequence.add_pose_frame(PoseFrame())

    trimmer = TrimPoseSequence(original_sequence)
    
    trimmed_sequence = trimmer.trim(start_trim=2, end_trim=3)

    print(f"Original length: {len(original_sequence)}")
    print(f"Trimmed length: {len(trimmed_sequence)}")
