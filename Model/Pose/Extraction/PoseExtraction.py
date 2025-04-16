# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

import cv2
from tqdm import tqdm
import numpy as np
import mediapipe as mp

from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark
from Model.Pose.Sequence.PoseSequence import PoseSequence


from Model.Utils.Checks import Check
from Model.Utils.MediaSteams import Video, Photo, MediaStream, Live
from Model.Utils.SpeedDecorators import monitor_speed
from Model.Utils.BaseClass import BaseClass



class BasePoseExtraction(BaseClass):
    def __init__(self, media_steam: MediaStream) -> None:
        """
        Base class for extracting pose sequences from a video.
        """
        self.media_steam = media_steam
        self.pose_sequence = PoseSequence(media_stream=media_steam)
        
        if type(media_steam) == Video:
            self.progress_bar = tqdm(total=self.media_steam.frame_count, desc=f'Analyzing Steam: "{self.media_steam.source}"')
        else:
            self.progress_bar = None
            
        
        self._extract_points()

    # ------------------------------ Pose Stuff ------------------------------ #
    def get_pose_sequence(self) -> PoseSequence:
        """Returns the extracted pose sequence."""
        return self.pose_sequence
    
    def _extract_points(self):
        """
        Processes the video frame by frame, extracting pose landmarks.
        """
        try:
            while True:
                ret, frame = self.media_steam.read()
                if not ret:
                    break


                landmarks = self._extract_points_from_frame(frame)
                pose_frame = PoseFrame(landmarks)
                self.pose_sequence.add_pose_frame(pose_frame)

                self._update_progress()
        except KeyboardInterrupt:
            pass

        self.media_steam.release()
        self._close_progress()

    # ------------------------------ Progress Bar Stuff ------------------------------ #

    def _update_progress(self):
        """Updates the progress bar by one step."""
        if self.progress_bar:
            self.progress_bar.update(1)

    def _close_progress(self):
        """Closes the progress bar."""
        if self.progress_bar:
            self.progress_bar.close()


class MediaPipePoseExtraction(BasePoseExtraction):
    mp_pose = mp.solutions.pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=True
    )

    def __init__(self, media_steam: MediaStream) -> None:
        """
        Extracts pose landmarks from a video using MediaPipe.
        """
        super().__init__(media_steam)


    def _extract_points_from_frame(self, frame: np.ndarray):
        """
        Converts a frame to RGB and runs MediaPipe pose estimation.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_pose.process(rgb_frame)
        
        if results and results.pose_landmarks:
            return self._convert_landmarks(results)
        
        return None

    def _convert_landmarks(self, pose_results) -> list[Landmark]:
        """
        Converts MediaPipe landmarks into a list of Landmark instances.
        """
        return [Landmark(lm.x, lm.y, lm.z, lm.visibility)
                for lm in pose_results.pose_landmarks.landmark]


    
    

if __name__ == "__main__":
    
    from Model.Pose.View.PersonViewer3D import PoseViewer3D
    # # Video Media
    video_path = "/Users/christopherarmenio/Desktop/IMG_1046.mov"
    video = Video(video_path)
    pose_extractor = MediaPipePoseExtraction(video)
    pose_sequence = pose_extractor.get_pose_sequence()
    viewer = PoseViewer3D(pose_sequence)
    viewer.show()
    
    
    # # # Photo Media
    # img_path = "Videos/IMG_1091.HEIC"
    # media = Photo(img_path)
    # pose_extractor = MediaPipePoseExtraction(media)
    # pose_sequence = pose_extractor.get_pose_sequence()
    # # pose_sequence.save("Sequences/pose_sequence_photo.pkl")
    # viewer = PoseViewer2D(pose_sequence, media)
    # viewer.show()
    
    # Live Media
    # camera_int = 0
    # media = Live(camera_int)
    # pose_extractor = MediaPipePoseExtraction(media)
    # pose_sequence = pose_extractor.get_pose_sequence()
    # pose_sequence.save("Sequences/pose_sequence_live.pkl")
    
    # viewer = PoseViewer3D(pose_sequence)
    # viewer.show()

    # Transform with person warp
    # video_path = "/Users/christopherarmenio/Desktop/IMG_1046.mov"
    # media = Video(video_path)
    # pose_extractor = MediaPipePoseExtraction(media)
    # pose_sequence = pose_extractor.get_pose_sequence()
    # pose_sequence.save('Sequences/IMG_1046.pkl')
    
    # pose_sequence = PoseSequence.load('Sequences/side_pose_sequence.pkl')
    
    # person = Person()
    # p_transformer = PoseTransformer(pose_sequence)
    # p_transformer.person_warp(person)
    # # pose_sequence.save("Sequences/pose_sequence_photo.pkl")
    # viewer = PoseViewer3D(pose_sequence)
    # viewer.show()