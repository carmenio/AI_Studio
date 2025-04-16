# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

import cv2
import numpy as np

from Model.Utils.BaseClass import BaseClass
import cv2
import numpy as np

from PIL import Image
from pi_heif import register_heif_opener

import tempfile
import moviepy.editor as mp
from moviepy.editor import VideoFileClip
import math

from typing import Optional,Union

import time


class MediaStream(BaseClass):
    def __init__(self, source):
        self.source = source

    def read(self):
        """Subclasses must implement read()."""
        raise NotImplementedError("Subclasses must implement read() method.")

    def play(self):
        """Subclasses must implement play()."""
        raise NotImplementedError("Subclasses must implement play() method.")
    
    def reset(self):...
    
    def release(self):...
    
class Video(MediaStream):
    def __init__(self, video_path: str, start: Union[int, float] = 0, end: Union[int, float, None] = None) -> None:
        super().__init__(video_path)
        self.video_path = video_path
        self.start = start
        self.end = end
        self.orientation_data_path = None
        
        self._init_capture()
        
    def _init_capture(self):
        self.cap = cv2.VideoCapture(self.video_path)  # Use composition, not inheritance
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {self.video_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.frame_count / self.fps

        self.rotation_code = self._get_rotation_code()
        self.start_frame = max(0, int(self.start * self.fps))
        self.end_frame = int(self.end * self.fps) if self.end else self.frame_count - 1
        self.end_frame = min(self.end_frame, self.frame_count - 1)
        self.current_frame = self.start_frame

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def _get_rotation_code(self):
        """Retrieves the video rotation metadata and returns the appropriate OpenCV rotation flag."""
        rotation_map = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE
        }
        metadata = cv2.VideoCapture(self.video_path)
        rotation = int(metadata.get(cv2.CAP_PROP_ORIENTATION_META)) if metadata.isOpened() else 0
        metadata.release()
        return rotation_map.get(rotation, None)

    def _correct_orientation(self, frame: np.ndarray) -> np.ndarray:
        """Corrects the orientation of a video frame if needed."""
        if self.rotation_code is not None:
            return cv2.rotate(frame, self.rotation_code)
        return frame

    def read(self):
        """Reads the next frame from the video."""
        if self.current_frame > self.end_frame:
            return False, None  

        ret, frame = self.cap.read()
        if not ret:
            return ret, None
        
        self.current_frame += 1
        frame = self._correct_orientation(frame)
        return ret, frame

    def play(self):
        """Plays the selected part of the video at the correct FPS."""
        frame_duration = 1 / self.fps

        while self.cap.isOpened():
            start_time = time.time()
            ret, frame = self.read()
            if not ret:
                break

            cv2.imshow("Video Playback", frame)
            elapsed_time = time.time() - start_time
            remaining_time = max(0, frame_duration - elapsed_time)
            time.sleep(remaining_time)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def reset(self):
        """Resets the video to the start frame."""
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __getstate__(self):
        """Define what gets pickled."""
        state = self.__dict__.copy()
        state["cap"] = None  # Exclude cv2.VideoCapture from pickling
        return state

    def __setstate__(self, state):
        """Define how to unpickle the object."""
        self.__dict__.update(state)
        self.cap = cv2.VideoCapture(self.video_path)  # Reinitialize VideoCapture

    def get_high_points(self, decibel_threshold: float, window_size: float = 0.5) -> list:
        """
        Process the video at video_path and return timestamps (in seconds)
        where the average decibel level over a window of window_size seconds
        exceeds the given threshold (in dB).

        Parameters:
        video_path (str): Path to the video file.
        threshold (float): Decibel threshold (default -25 dB).
        window_size (float): Duration (in seconds) of each analysis window.

        Returns:
        list of float: Timestamps (in seconds) where the window's dB level > threshold.
        """
        # Load the video and extract its audio
        clip = VideoFileClip(self.video_path)
        audio = clip.audio
        sample_rate = audio.fps  # sampling rate in Hz

        # Explicitly pass a chunksize (here, one second worth of samples) to iter_chunks.
        chunksize = sample_rate  # one second of audio samples
        chunks = list(audio.iter_chunks(fps=sample_rate, chunksize=chunksize))
        samples = np.vstack(chunks)

        # If stereo, average the channels to get a mono signal.
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        # Determine the number of samples per window.
        window_samples = int(window_size * sample_rate)
        n_windows = len(samples) // window_samples  # drop any remainder

        # Truncate samples to have a whole number of windows and reshape.
        samples = samples[:n_windows * window_samples]
        windows = samples.reshape((n_windows, window_samples))

        # Compute the RMS (root mean square) for each window and convert to dB.
        rms = np.sqrt(np.mean(windows**2, axis=1))
        dB = 20 * np.log10(rms + 1e-10)  # small epsilon to avoid log10(0)

        # Find the indices of windows where dB exceeds the threshold.
        indices = np.where(dB > decibel_threshold)[0]
        loud_points = indices * window_size

        clip.close()
        return loud_points.tolist()

    def split_video(self, start_time: float, end_time: float) -> "Video":
        """
        Splits the video between the specified start and end times and returns a new Video instance
        containing only that segment.

        Parameters:
            start_time (float): The starting timestamp in seconds.
            end_time (float): The ending timestamp in seconds.

        Returns:
            Video: A new Video object representing the subclip.
        """
        # Calculate the full video duration using the frame count and fps
        return Video(self.video_path, start=start_time, end=end_time)
    
    def add_orientation_data(self, orientation_data_path):
        self = OrientatedVideo(self.video_path, orientation_data_path, start=self.start, end=self.end)
        
    
        
class OrientatedVideo(Video):
    def __init__(self, video_path: str, orientation_data_path:str, start: Union[int, float] = 0, end: Union[int, float, None] = None) -> None:
        super().__init__(video_path, start, end)
        self.orientation_data_path = orientation_data_path
        
        
    # def play(self):
    #     """Plays the selected part of the video at the correct FPS."""
    #     frame_duration = 1 / self.fps

    #     while self.cap.isOpened():
    #         start_time = time.time()
    #         ret, frame = self.read()
    #         if not ret:
    #             break

    #         cv2.imshow("Video Playback", frame)
    #         elapsed_time = time.time() - start_time
    #         remaining_time = max(0, frame_duration - elapsed_time)
    #         time.sleep(remaining_time)

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break

    #     self.cap.release()
    #     cv2.destroyAllWindows()
        
        
        

    

class Live(MediaStream):
    def __init__(self, device_index=0) -> None:
        """
        For a live stream, 'device_index' can be an integer (e.g., 0 for the default webcam)
        or even a stream URL.
        """
        super().__init__(device_index)
        self.device_index = device_index  # Store device index for reinitialization
        self._cap = cv2.VideoCapture(device_index)  # Use an instance variable instead
        
        if not self._cap.isOpened():
            raise ValueError(f"Unable to open live video stream: {device_index}")
        
        self.fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        self.rotation_code = None  

    def read(self):
        """Reads a frame from the video stream and handles KeyboardInterrupt."""
        try:
            ret, frame = self._cap.read()
            return ret, frame
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected. Stopping stream...")
            self._cap.release()
            cv2.destroyAllWindows()
            

    def play(self):
        """Plays the live video stream."""
        while self._cap.isOpened():
            ret, frame = self.read()
            if not ret:
                break
            
            cv2.imshow("Live Stream", frame)
            
            if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to exit
                break

        self._cap.release()
        cv2.destroyAllWindows()

    def __getstate__(self):
        """Custom method to remove unpicklable objects before pickling."""
        state = self.__dict__.copy()
        del state["_cap"]  # Remove cv2.VideoCapture object
        return state

    def __setstate__(self, state):
        """Custom method to restore unpickled object and reinitialize VideoCapture."""
        self.__dict__.update(state)
        self._cap = cv2.VideoCapture(self.device_index)  # Reinitialize camera
        
        
    def release(self):...



class Photo(MediaStream):
    def __init__(self, image_path: str) -> None:
        super().__init__(image_path)
        
        self.has_been_read = False

        # Check if the image is a HEIC file based on the file extension
        if image_path.lower().endswith('.heic'):
            self.image = self._load_heic_image(image_path)
        else:
            self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Unable to open image file: {image_path}")

    def _load_heic_image(self, image_path: str) -> np.ndarray:
        register_heif_opener()
        """Loads a HEIC image using Pillow (with pillow-heif plugin) and converts it to a format compatible with OpenCV."""
        image = Image.open(image_path)  # Load HEIC image using Pillow
        image = np.array(image)  # Convert Pillow image to NumPy array
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV compatibility
        return image

    def read(self):
        """
        For a photo, there's only one image to return.
        """
        if self.has_been_read:
            return False, None  # No more frames to read
        else:
            self.has_been_read = True  # Mark it as read
            return True, self.image
            
    def play(self):
        """Displays the photo."""
        cv2.imshow("Photo", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    def release(self):...
    
    def reset(self):
        self.has_been_read = False


if __name__ == "__main__":
    # video = Video("/Users/christopherarmenio/Desktop/Golf Dataset/Dataset/Videos_Broken_Up/Back/segment_1.mp4", start=0, end=3)
    
    video = OrientatedVideo('Videos/iPhoneRecordings/Vide_2/recording_1741669263.9204202.mov', 'Videos/iPhoneRecordings/Vide_2/orientation_1741669263.9204202.json')
    video.play()
    # print(
    #     video.get_high_points(-25.0, 0.1)
    # )
    # subVideo = video.split_video(10.1,20.1)
    # video.play()
    
    # print(video.get_high_points(-30))
    # print(video.fps)
    # print(video.frame_count)
    # video.play()
    
    
    # video = Photo("/Users/christopherarmenio/Downloads/IMG_1082.HEIC")
    # video.play()
    
    
    # video = Live(0)
    # video.play()
