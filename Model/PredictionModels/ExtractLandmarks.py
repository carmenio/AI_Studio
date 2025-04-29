# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import PoseFrame
from Model.Pose.Sequence.PoseSequence import PoseSequence
from typing import List

class ExtractLandmarks:
    """
        Extract Landmarks corresponding to a particular type of gait.
    """

    def __init__(self, gait_type: str, gait_type_path: str):
        """
        Initialise the ExtractLandmarks class.

        Args:
            gait_type (str, optional): The type of gait to be processed.
            gait_type_path (str, optional): The path to the gait type pkl data.

        Attributes:
            landmarks (dict): A dictionary to store extracted landmarks.
            timestep_ (int): Holds the total number of timesteps for this gait type.
        """
        self.landmarks = {}
        self.type = gait_type
        self.gait_type_path = gait_type_path
        self.timestep_ = 0
        self.video_mapping = {}  # Maps timestamps to source video IDs
        self.sequence_mapping = {}  # Maps sequence objects to video IDs

    def load_pkl_file(self, filepath: str):
        """
            Return a sequence of pose landmarks. 
        """
        return PoseSequence.load(filepath)
    
    # def load_pkl_files(self, filepaths: List[str]):
    #     """
    #         Return a list containing pose sequences corresponding to number of pkl files relevant to that gait.
    #     """
        
    #     # self.pose_sequences = [self.load_pkl_file(file) for file in filepaths]
    #     self.pose_sequences = {} # Dictionary to hold pose sequences with their respective file names as keys
    #     for filepath in filepaths:
    #         # Extract the file name without the extension
    #         filename = os.path.splitext(os.path.basename(filepath))[0] # "0.pkl" -> "0"
    #         # Load the pose sequence and store it in the dictionary
    #         self.pose_sequences[filename] = self.load_pkl_file(filepath)

    def load_pkl_files(self, filepaths: List[str]):
        """
            Load multiple PKL files and map them to video IDs based on filename.
        """
        self.pose_sequences = []
        self.sequence_mapping = {}
        
        for file in filepaths:
            # Extract video ID from filename 
            video_id = os.path.basename(file).split('.')[0]  # Gets filename without extension
            
            # Load the sequence
            sequence = self.load_pkl_file(file)
            
            # Store the sequence
            self.pose_sequences.append(sequence)
            
            # Map the sequence object to its video ID
            self.sequence_mapping[sequence] = video_id

    # # NOTE: You dont need to extract all the landmarks, you can just index the object like a normal array.
    # # eg. self.pose_sequences[0] will give you all the landmarks at frame 0
    # def extract_single_seq(self, pose_seq):
    #     """
    #         Extract landmarks corresponding to a single pose sequence and add them to the landmarks dict.
    #     """

    #     for frame in pose_seq:
    #         self.timestep_ += 1
    #         frame : PoseFrame
    
    #         self.landmarks[self.timestep_] = frame.get_landmarks()

    def extract_single_seq(self, pose_seq):
        """
            Extract landmarks corresponding to a single pose sequence and add them to the landmarks dict.
            Also tracks which video each frame comes from.
        """
        # Get the video ID for this sequence
        video_id = self.sequence_mapping.get(pose_seq)
        
        for frame in pose_seq:
            self.timestep_ += 1
            frame : PoseFrame
            
            # Store the landmarks
            self.landmarks[self.timestep_] = frame.get_landmarks()
            
            # Also store which video this frame came from
            self.video_mapping[self.timestep_] = video_id

    def extract(self):
        """
            Extract landmarks for all sequences. 
        """
        # NOTE: Here you are going through each frame not seq
        for seq in self.pose_sequences:
            self.extract_single_seq(seq)


if __name__ == "__main__":
    GAIT_PATH = "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait"
    extractor = ExtractLandmarks("Normal_Gait", GAIT_PATH)

    pkl_files = sorted([os.path.join(GAIT_PATH, file) for file in os.listdir(GAIT_PATH) if file.endswith('.pkl')])

    # print(pkl_files)  # Print the list of pkl files
    # Load the pkl files into the extractor
    extractor.load_pkl_files(pkl_files)
    # Load the pose sequences from the pkl files
    extractor.extract()
    # print(extractor.landmarks)  # Print the extracted landmarks
    for timestep in sorted(list(extractor.landmarks.keys()))[:300]:  # Show first 200 frames
        print(f"Frame {timestep} from video {extractor.video_mapping[timestep]}")

    
