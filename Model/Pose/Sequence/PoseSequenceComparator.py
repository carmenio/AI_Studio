"""
PoseSequenceComparator.py

Provides a class for comparing two PoseSequence objects and returning
a normalized difference score. Normalization is done per-frame based on
a reference distance (default: shoulder width) to account for different
body sizes.
"""
# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

import numpy as np
from typing import Tuple
from Model.Pose.Edit.Trim import TrimPoseSequence

from Model.Pose.Sequence.PoseSequence import PoseSequence
from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark

class PoseSequenceComparator:
    """
    Compare two PoseSequence instances and compute a normalized difference score.
    """
    def __init__(self, reference_landmarks: Tuple[int, int] = (11, 12)):
        """
        :param reference_landmarks: A tuple of two landmark indices to define
                                    the scale (e.g., left and right shoulder).
        """
        self.ref_idx_a, self.ref_idx_b = reference_landmarks

    def _frame_scale(self, frame: PoseFrame) -> float:
        """
        Compute a scale factor for a single frame based on the distance
        between two reference landmarks.
        """
        lm_a: Landmark = frame[self.ref_idx_a]
        lm_b: Landmark = frame[self.ref_idx_b]
        # Euclidean distance in 3D
        return float(np.linalg.norm(np.array([lm_a.x, lm_a.y, lm_a.z]) -
                                    np.array([lm_b.x, lm_b.y, lm_b.z])))

    def _normalize_frame(self, frame: PoseFrame) -> np.ndarray:
        """
        Extract landmark coordinates and normalize them by the frame scale.
        Returns an array of shape (num_landmarks, 3).
        """
        scale = self._frame_scale(frame)
        if scale <= 0:
            raise ValueError(f"Invalid scale ({scale}) in frame; cannot normalize.")
        coords = np.array([[lm.x, lm.y, lm.z] for lm in frame.landmarks], dtype=float)
        return coords / scale

    def _normalize_sequence(self, sequence: PoseSequence) -> np.ndarray:
        """
        Convert and normalize an entire PoseSequence.
        Returns an array of shape (num_frames, num_landmarks, 3).
        """
        normalized = []
        for frame in sequence:
            normalized.append(self._normalize_frame(frame))
        return np.stack(normalized, axis=0)

    # NOTE: there may be an issue with the trimming, be cause i trim from the end
    #       how ever the prediction doesn't work until frame 1 or 5 i think, so may need a buffer.
    def compare(self, seq_a: PoseSequence, seq_b: PoseSequence) -> float:
        """
        Compare two sequences and return the average normalized difference score.
        Sequences are first trimmed equally from start and end to match lengths, then normalized.
        """
        # Trim the longer sequence equally from start and end so both have equal length
        len_a, len_b = len(seq_a), len(seq_b)
        if len_a != len_b:
            diff = len_a - len_b
            if diff > 0:
                # seq_a is longer
                start_trim = diff // 2
                end_trim = diff - start_trim
                seq_a = TrimPoseSequence(seq_a).trim(start_trim=start_trim, end_trim=end_trim)
            else:
                # seq_b is longer
                diff = -diff
                start_trim = diff // 2
                end_trim = diff - start_trim
                seq_b = TrimPoseSequence(seq_b).trim(start_trim=start_trim, end_trim=end_trim)
                
        # Normalize both sequences
        norm_a = self._normalize_sequence(seq_a)
        norm_b = self._normalize_sequence(seq_b)
        # Check for NaNs in normalized data
        if np.isnan(norm_a).any():
            raise ValueError(f"NaNs found in normalized sequence A (count={np.isnan(norm_a).sum()})")
        if np.isnan(norm_b).any():
            raise ValueError(f"NaNs found in normalized sequence B (count={np.isnan(norm_b).sum()})")
        
        

        # Align on shortest length
        min_frames = min(norm_a.shape[0], norm_b.shape[0])
        if min_frames == 0:
            raise ValueError("One or both sequences are empty.")

        # Trim to common length
        a_trim = norm_a[:min_frames]
        b_trim = norm_b[:min_frames]

        # Compute per-landmark, per-frame distances
        diffs = a_trim - b_trim
        dists = np.linalg.norm(diffs, axis=2)  # shape (min_frames, num_landmarks)

        # Average over frames and landmarks
        return float(np.mean(dists))
    
    
if __name__ == "__main__":
    from Model.Pose.Waveform.PoseWaveform import PoseWaveforms
    ground_truth = PoseSequence.load("Sequences/Apr_10_Antalgic_Gait.pkl")

    seq_mediapipe = PoseSequence.load("Sequences/Apr_10_Antalgic_Gait_mediapipe.pkl")
    seq_mediapipe = PoseWaveforms(seq_mediapipe).get_sequence()
    
    
    seq_predicted = PoseSequence.load("Sequences/Apr_10_Antalgic_Gait_predicted.pkl")
    seq_predicted = PoseWaveforms(seq_predicted).get_sequence()


    seq_open_pose = PoseSequence.load("Sequences/Apr_10_Antalgic_Gait_open_pose.pkl")
    seq_open_pose = PoseWaveforms(seq_open_pose).get_sequence()

    comparator = PoseSequenceComparator()
    score = comparator.compare(ground_truth, seq_mediapipe)
    print(f"Normalized difference score: {score:.4f} (Media Pipe)")

    comparator = PoseSequenceComparator()
    score = comparator.compare(ground_truth, seq_predicted)
    print(f"Normalized difference score: {score:.4f} (Custom Model)")

    comparator = PoseSequenceComparator()
    score = comparator.compare(ground_truth, seq_open_pose)
    print(f"Normalized difference score: {score:.4f} (Open Pose)")
