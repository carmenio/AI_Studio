import os, sys
[(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) 
                                  for i in range(len(os.getcwd().split(os.sep))))
 if os.path.isfile(os.path.join(d, 'main.py'))]

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, RadioButtons
import matplotlib.patches as mpatches

from Model.Pose.Sequence.PoseSequence import PoseSequence
# FIXME: Needs to be removed later, keep here for now
from Model.Pose.Sequence.PoseSequence import PoseSequence as PoseWaveforms
from Model.Utils.BaseClass import BaseClass

from typing import Union

# ---------------------------------------------------------------------
# Base Viewer with common plotting methods
# ---------------------------------------------------------------------
class Viewer(BaseClass):
    _CONNECTIONS = [
        (11, 13), (13, 15),  # Right arm
        (12, 14), (14, 16),  # Left arm
        (11, 12),            # Shoulders
        (23, 24), (23, 11),  # Torso connections
        (24, 12), (23, 25),  # Hip connections
        (25, 27), (24, 26),  # Legs
        (26, 28), (27, 29),  # Lower legs
        (29, 31), (28, 30),  # Feet
        (30, 32), (0, 1),    # Feet and face
        (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8),
        (9, 10)
    ]

    def show(self):
        skeleton_data = np.array(self.get_skeleton_points())
        num_frames = len(skeleton_data)

        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.1, top=0.85)
        
        # Setup 3D axis and widgets
        ax_3d = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax_3d)
        
        slider, view_radio = self._create_widgets(fig, num_frames)
        scatter, lines = self._initialize_visual_elements(ax_3d)

        # Initialize view parameters
        self.current_azim = -60
        self.current_elev = 30

        def update_view(label):
            self._handle_view_change(label, ax_3d)
            fig.canvas.draw_idle()

        def update(val):
            self._update_frame(
                int(val), skeleton_data, 
                ax_3d, scatter, lines, slider
            )
            fig.canvas.draw_idle()

        slider.on_changed(update)
        view_radio.on_clicked(update_view)
        update(0)
        plt.show()

    def _setup_axes(self, ax):
        ax.set_xlim(0, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(1, 0)
        ax.set_xlabel('X')
        ax.set_ylabel('Depth (Y)')
        ax.set_zlabel('Height (Z)')

    def _create_widgets(self, fig, num_frames):
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])
        ax_radio = fig.add_axes([0.1, 0.92, 0.2, 0.1])
        return (
            Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1),
            RadioButtons(ax_radio, ['Default', 'Front (Y)', 'Side (X)', 'Top (Z)'])
        )

    def _initialize_visual_elements(self, ax):
        # Use the viewer's color for the markers and lines
        scatter = ax.plot([], [], [], marker='o', markersize=5, color=self.dot_color, linestyle='')[0]
        lines = [ax.plot([], [], [], color=self.color, linestyle='-')[0] for _ in self._CONNECTIONS]
        return scatter, lines

    def _handle_view_change(self, label, ax):
        views = {
            'Front (Y)': (90, 0),
            'Side (X)': (0, 0),
            'Top (Z)': (0, 90),
            'Default': (-60, 30)
        }
        self.current_azim, self.current_elev = views[label]
        ax.view_init(elev=self.current_elev, azim=self.current_azim)

    def _update_frame(self, frame_idx, data, ax, scatter, lines, slider):
        # Extract coordinates
        frame_points = data[frame_idx]
        x = [landmark.x for landmark in frame_points]
        y = [landmark.y for landmark in frame_points]
        z = [landmark.z for landmark in frame_points]
        
        # Update scatter plot
        scatter.set_data(x, z)
        scatter.set_3d_properties(y)

        # Update connection lines
        for i, (start, end) in enumerate(self._CONNECTIONS):
            lines[i].set_data([x[start], x[end]], [z[start], z[end]])
            lines[i].set_3d_properties([y[start], y[end]])

        ax.set_title(f"Frame {frame_idx}")
        slider.valtext.set_text(str(frame_idx))

    def save_video(self, direction: str, output_filename: str, fps: int = 30):
        skeleton_data = np.array(self.get_skeleton_points())
        num_frames = len(skeleton_data)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)

        view_mapping = {
            'Front': 'Front (Y)',
            'Side': 'Side (X)',
            'Top': 'Top (Z)',
            'Default': 'Default'
        }
        view_label = view_mapping.get(direction, 'Default')
        self._handle_view_change(view_label, ax)

        scatter, lines = self._initialize_visual_elements(ax)

        def update(frame_idx):
            frame_points = skeleton_data[frame_idx]
            x = [landmark.x for landmark in frame_points]
            y = [landmark.y for landmark in frame_points]
            z = [landmark.z for landmark in frame_points]
            
            scatter.set_data(x, z)
            scatter.set_3d_properties(y)
            
            for i, (start, end) in enumerate(self._CONNECTIONS):
                lines[i].set_data([x[start], x[end]], [z[start], z[end]])
                lines[i].set_3d_properties([y[start], y[end]])
            
            ax.set_title(f"Frame {frame_idx}")
            return scatter, *lines

        anim = FuncAnimation(fig, update, frames=num_frames, interval=1000/fps, blit=False)
        writer = FFMpegWriter(fps=fps)
        anim.save(output_filename, writer=writer)
        plt.close(fig)

# ---------------------------------------------------------------------
# PoseViewer3D: single skeleton viewer (with custom color and label)
# ---------------------------------------------------------------------
class PoseViewer3D(Viewer):
    def __init__(self, seq: Union[PoseSequence,PoseWaveforms], color='blue', dot_color='grey', label='Skeleton'):
        self.seq = seq
        self.color = color
        self.dot_color = dot_color
        self.label = label

    def get_skeleton_points(self):
        return [frame.get_landmarks() for frame in self.seq.get_sequence()]

    def __add__(self, other):
        if isinstance(other, PoseViewer3D):
            return MultiPoseViewer3D([self, other])
        else:
            return NotImplemented

# ---------------------------------------------------------------------
# MultiPoseViewer3D: displays multiple skeletons with individual colors
# and a legend key.
# ---------------------------------------------------------------------
class MultiPoseViewer3D(Viewer):
    def __init__(self, viewers):
        """
        viewers: list of PoseViewer3D instances.
        """
        self.viewers = viewers
        # Get the skeleton data for each viewer
        self.skeletons_data = [viewer.get_skeleton_points() for viewer in viewers]
        # Use the minimum number of frames among all sequences
        self.num_frames = min(len(data) for data in self.skeletons_data)

    def __add__(self, other):
        if isinstance(other, PoseViewer3D):
            return MultiPoseViewer3D(self.viewers + [other])
        elif isinstance(other, MultiPoseViewer3D):
            return MultiPoseViewer3D(self.viewers + other.viewers)
        else:
            return NotImplemented
        
    def show(self):
        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.1, top=0.85)
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)

        slider, view_radio = self._create_widgets(fig, self.num_frames)
        
        # Create visual elements for each viewer
        visuals = []  # List of (scatter, lines) for each viewer
        for viewer in self.viewers:
            scatter, lines = viewer._initialize_visual_elements(ax)
            visuals.append((scatter, lines))
        
        # Add a legend indicating which color corresponds to which skeleton
        
        handles = [mpatches.Patch(color=viewer.color, label=viewer.label) for viewer in self.viewers]
        ax.legend(handles=handles)

        # Initialize view parameters
        current_azim = -60
        current_elev = 30
        ax.view_init(elev=current_elev, azim=current_azim)

        def update_view(label):
            views = {
                'Front (Y)': (90, 0),
                'Side (X)': (0, 0),
                'Top (Z)': (0, 90),
                'Default': (-60, 30)
            }
            azim, elev = views[label]
            ax.view_init(elev=elev, azim=azim)
            fig.canvas.draw_idle()

        def update(val):
            frame_idx = int(val)
            for idx, viewer in enumerate(self.viewers):
                # For safety, use the viewer's data if available.
                data = self.skeletons_data[idx][frame_idx]
                x = [landmark.x for landmark in data]
                y = [landmark.y for landmark in data]
                z = [landmark.z for landmark in data]
                scatter, lines = visuals[idx]
                scatter.set_data(x, z)
                scatter.set_3d_properties(y)
                for i, (start, end) in enumerate(viewer._CONNECTIONS):
                    lines[i].set_data([x[start], x[end]], [z[start], z[end]])
                    lines[i].set_3d_properties([y[start], y[end]])
            ax.set_title(f"Frame {frame_idx}")
            slider.valtext.set_text(str(frame_idx))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        view_radio.on_clicked(update_view)
        update(0)
        plt.show()


# ---------------------------------------------------------------------
# Example usage in main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from Model.Pose.Extraction.PoseExtraction import MediaPipePoseExtraction
    from Model.Utils.MediaSteams import Video
    from Model.Pose.Person.Person import Person
    # from test import LimbRatioAnalyzer, Transformer
    from Model.Pose.Waveform.PoseWaveform import PoseWaveforms
    
    
    # vid = Video('/Users/christopherarmenio/Downloads/IMG_1081.mov')
    # pe = MediaPipePoseExtraction(vid)
    # seq = pe.get_pose_sequence()
    # seq.save('Sequences/Chris_Kitchen_Front.pkl')
    
    
    # vid2 = Video('/Users/christopherarmenio/Downloads/IMG_1143.mov')
    # pe2 = MediaPipePoseExtraction(vid2)
    # seq2 = pe2.get_pose_sequence()
    # seq2.save('Sequences/Chris_Kitchen_Side.pkl')
    
    seq = PoseSequence.load('Sequences/Chris_Kitchen_Front.pkl')
    waveform = PoseWaveforms(seq, set_interval=0.001)
    
    viewer = PoseViewer3D(waveform, color='blue', label='Front Pose')
    viewer.show()
    
    # seq1 = PoseSequence.load('Sequences/Chris_Kitchen_Front.pkl')
    # seq2 = PoseSequence.load('Sequences/Chris_Kitchen_Side.pkl')
    # seq3 = PoseSequence.load('Sequences/front_pose_sequence.pkl')
    
    # pt = PoseTransformer(seq2)
    # pt.rotate_body(y_rotation=90)
    
    # # analyzer = LimbRatioAnalyzer(seq, alpha=0.1, beta=1.0, confidence_threshold=0.5)

    # # # Run the analysis pipeline
    # # person: Person = analyzer.get_person()
    
    # # transformer = Transformer(person)
    # # new_seq = transformer.warp_pose_sequence(seq)
    
    # # # Create two viewers with different colors and labels.
    # viewer1 = PoseViewer3D(seq1, color='blue', label='Front Pose')
    # viewer2 = PoseViewer3D(seq2, color='red', label='Side Pose')
    # viewer3 = PoseViewer3D(seq3, color='pink', label='random')
    # # viewer3.show()
    # # # # For demonstration, you could create a transformed/different sequence.
    # # # # Here we use the same sequence, but you might use PoseTransformer, etc.
    
    
    # # # other_vid = Video('/Users/christopherarmenio/Desktop/IMG_1143.mov')
    # # # other_pe = MediaPipePoseExtraction(other_vid)
    # # # other_seq = other_pe.get_pose_sequence()
    # # viewer2 = PoseViewer3D(new_seq, color='red', label='Modified Pose')
    
    # # # Combine the two viewers using the overloaded '+' operator.
    # combined_viewer = viewer2 + viewer1 + viewer3
    # combined_viewer.show()
