# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark
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

    def load_pkl_file(self, filepath: str):
        """
            Return a sequence of pose landmarks. 
        """
        return PoseSequence.load(filepath)
    
    def load_pkl_files(self, filepaths: List[str]):
        """
            Return a list containing pose sequences corresponding to number of pkl files relevant to that gait.
        """

        self.pose_sequences = [self.load_pkl_file(file) for file in filepaths]
    
    # NOTE: You dont need to extract all the landmarks, you can just index the object like a normal array.
    # eg. self.pose_sequences[0] will give you all the landmarks at frame 0
    def extract_single_seq(self, pose_seq):
        """
            Extract landmarks corresponding to a single pose sequence and add them to the landmarks dict.
        """

        for frame in pose_seq:
            self.timestep_ += 1
            # landmarks_list = []
            frame : PoseFrame
            # for landmark in frame:
            #     landmark : Landmark
            #     landmarks_list.append(landmark)

            self.landmarks[self.timestep_] = frame.get_landmarks()

    def extract(self):
        """
            Extract landmarks for all sequences. 
        """
        # NOTE: Here you are going through each frame not seq
        for seq in self.pose_sequences:
            self.extract_single_seq(seq)


if __name__ == "__main__":
    GAIT_PATH = "Model/PredictionModels/Sequences/Apr_10_Antalgic_Gait"
    extractor = ExtractLandmarks("Antalgic_Gait", GAIT_PATH)

    extractor.load_pkl_files(sorted(os.listdir(GAIT_PATH)))
    extractor.extract()
    print(extractor.landmarks)  # Print the extracted landmarks

    
