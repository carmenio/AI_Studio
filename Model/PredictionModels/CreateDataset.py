# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

from Model.PredictionModels.ExtractLandmarks import ExtractLandmarks
from Model.PredictionModels.GaitMetrics import GaitMetrics
from typing import List
import pandas as pd

class Dataset:
    def __init__(self, rha: List[int], lha: List[int], rka: List[int], lka: List[int], bta: List[int], hta: List[int]):
        # init empty dataframe
        self.final_df = pd.DataFrame()
        self.rha = rha
        self.lha = lha  
        self.rka = rka
        self.lka = lka
        self.bta = bta
        self.hta = hta

    def dataset_single_gait_type(self, gait_type: str, gait_type_path: str):
        extractor = ExtractLandmarks(gait_type, gait_type_path) # D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait

        # Get the list of pkl files in the gait type path
        pkl_files = sorted([os.path.join(gait_type_path, file) for file in os.listdir(gait_type_path) if file.endswith('.pkl')])

        # store num of videos in the gait type
        n_videos = len(pkl_files)

        # Load the pkl files into the extractor
        extractor.load_pkl_files(pkl_files)

        # Load the pose sequences from the pkl files
        extractor.extract()

        gait_metrics = GaitMetrics(extractor.landmarks, extractor.video_mapping, gait_type)

        # * Inspect for each video of each gait type before entering the buffer value
        self.find_optim_buffer(gait_metrics, n_videos)

        # create the metrics DataFrame
        self.create_metrics_df(gait_metrics, self.rha, self.lha, self.rka, self.lka, self.bta, self.hta)

    # def find_optim_buffer(self, GM: GaitMetrics, n_videos: int):
    #     """
    #         Find the optimal buffer value for the gait metrics.
    #     """
    #     # loop through n_videos times and check the buffer value for each video.
    #     # (make sure to loop from 0 to n_videos - 1 in order)
    #     for i in range(n_videos):
    #         GM.find_optimal_buffer(29, i) # limitation - calculated only for left feet (Assuming there is no major difference between left and right feet)

    def find_optim_buffer(self, GM: GaitMetrics, n_videos: int):
        """
        Find the optimal buffer value for all videos of a gait type.
        """
        # Get unique video IDs (don't assume they are 0,1,2...n)
        unique_video_ids = sorted(set(GM.video_mapping.values()))
        buffer_values = []
        
        print(f"Finding optimal buffer values for {len(unique_video_ids)} videos...")
        
        # Collect buffer values for each video
        for vid_id in unique_video_ids:
            print(f"Processing video {vid_id}...")
            buffer_value = GM.find_optimal_buffer(29, vid_id)
            buffer_values.append(buffer_value)
            print(f"Video {vid_id} optimal buffer: {buffer_value}")
        
        # Calculate mean buffer value for this gait type
        if buffer_values:
            mean_buffer = sum(buffer_values) / len(buffer_values)
            print(f"Mean buffer value: {mean_buffer}")
            
            # Set this as the best buffer for the GaitMetrics instance
            GM.best_buffer = mean_buffer
        else:
            print("Warning: No buffer values collected")
            GM.best_buffer = 0.5  # Default fallback
    
    def create_metrics_df(self, GM: GaitMetrics, rhp: List[int], lhp: List[int], rka: List[int], lka: List[int], bta: List[int], hta: List[int]):
        """
            Create a DataFrame with the gait metrics.
        """
        # Create the DataFrame
        df = GM.create_metrics_df(
            rightHipAngle_req_landmarks=rhp, 
            leftHipAngle_req_landmarks=lhp,
            rightKneeAngle_req_landmarks=rka, 
            leftKneeAngle_req_landmarks=lka,
            bodyTiltAngle_req_landmarks=bta, 
            headTiltAngle_req_landmarks=hta
        )
        
        # Add the DataFrame to the final DataFrame
        self.final_df = pd.concat([self.final_df, df], ignore_index=True)

    def complete_dataset(self, gait_types: List[str], gait_type_paths: List[str]):
        """
            Create a complete dataset with all the gait types.
        """
        # loop through the gait types and create the dataset for each one
        for i in range(len(gait_types)):
            self.dataset_single_gait_type(gait_types[i], gait_type_paths[i])

        # save the DataFrame to a csv file
        self.final_df.to_csv("D:/AI_Studio/Model/PredictionModels/Sequences/FinalData.csv", index=False)
        

# example
if __name__ == "__main__":
    # Set the paths for the gait types
    gait_types = ["Antalgic_Gait", "Ataxic_Gait", "Circumduction_Gait", "Normal_Gait", "Scissoring_Gait", "Spastic_Gait"]
    gait_type_paths = [
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Antalgic_Gait",
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Ataxic_Gait",
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Circumduction_Gait",
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Normal_Gait",
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Scissoring_Gait",
        "D:/AI_Studio/Model/PredictionModels/Sequences/Apr_10_Spastic_Gait"
    ]
    
    # Create the dataset object
    dataset = Dataset(
        rha = [12, 24, 26],
        lha = [11, 23, 25],
        rka = [24, 26, 28],
        lka = [23, 25, 27],
        bta = [11, 12, 23, 24],
        hta = [0, 11, 12]
    )

    # Create the complete dataset
    dataset.complete_dataset(gait_types, gait_type_paths)

    