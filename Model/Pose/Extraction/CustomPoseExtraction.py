# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

import numpy as np
import cv2

from typing import List
from Model.Pose.Extraction.PoseExtraction import MediaPipePoseExtraction
from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark

from Model.Utils.SpeedDecorators import monitor_speed, monitor_culm_speed

class CustomMediaPipePoseExtraction(MediaPipePoseExtraction):
    
    def _extract_points_from_frame(self, frame: np.ndarray):
        """
        Converts a frame to RGB and runs MediaPipe pose estimation.
        """
        
        media_pipe_points = self._extract_2D_points(frame)
        landmark_points = self._convert_landmarks(media_pipe_points)
        
        new_landmark_points = self._extract_3D_points(landmark_points)
        
        return new_landmark_points
        
        
    def _extract_2D_points(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_frame)
        
        if not results or not results.pose_landmarks:
            return None
        
        # 
        return results 
        
    def _extract_3D_points(self, landmarks: List[Landmark]):
        ...
        
        # pass info into neural network 
        # info = {
        #     xy cord of each points,
        #     body measurements of the person
        #     velocity of points?
        #     acceleration of points?
        #     
        # }
        

if __name__ == "__main__":
    
    from Model.Pose.View.PersonViewer3D import PoseViewer3D
    from Model.Utils.MediaSteams import Video, Photo, MediaStream, Live
    
    # # Video Media
    video_path = "/Users/christopherarmenio/Desktop/IMG_1046.mov"
    video = Video(video_path)
    pose_extractor = CustomMediaPipePoseExtraction(video)
    pose_sequence = pose_extractor.get_pose_sequence()
    viewer = PoseViewer3D(pose_sequence)
    viewer.show()