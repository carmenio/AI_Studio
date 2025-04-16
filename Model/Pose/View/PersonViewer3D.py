# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d),) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]

# Import necessary libraries and modules
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.widgets import Slider, RadioButtons, Button
import matplotlib.patches as mpatches

# Import custom modules for pose sequence processing and visualization
from Model.Pose.Sequence.PoseSequence import PoseSequence
from Model.Utils.BaseClass import BaseClass

from typing import Union

# ------------------------------------------------------------------------------
# Base Viewer with common plotting methods for 3D pose visualization
# ------------------------------------------------------------------------------
class Viewer(BaseClass):
    """
    A base viewer class to handle the visualization of 3D pose data.
    It provides common functionality for setting up the plot, handling widgets,
    updating frames, and saving the visualization as a video.
    """
    # Define skeletal connections: pairs of indices corresponding to joints to be connected by lines.
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

    # Preset views (azimuth, elevation) for the 3D plot
    views = {
        'Front':   (-90, 0),
        'Side':    (0, 0),
        'Top':     (-90, 90),
        'Default': (-45, 35)
    }
    
    def __init__(self) -> None:
        """
        Initialize the viewer with default axis limits and colors.
        """
        super().__init__()
        # Default axis limits
        self.x_axes, self.y_axes, self.z_axes = 0, 0, 0 
        # Colors for the dots and skeleton lines
        self.dot_color = 'blue'
        self.color = 'grey'
        
        # Attributes for play functionality
        self.is_playing = False
        self.timer = None
        self.play_button = None
    
    def show(self):
        """
        Prepare the skeleton data, setup the plot and widgets,
        initialize the view, and display the interactive plot.
        """
        # Convert skeleton points to a numpy array
        self.skeleton_data = np.array(self.get_skeleton_points())
        self.num_frames = len(self.skeleton_data)
        self.setup_plot()
        # Initialize visual elements (scatter points and skeletal lines)
        self.scatter, self.lines = self._initialize_visual_elements(self.ax_3d)
        # Set up initial view parameters
        self.initialize_view_parameters()
        plt.show()
       
    # ------------------------------ Setup Methods ------------------------------ #
    def setup_plot(self):
        """
        Create the figure and 3D axis, adjust layout, and create widgets.
        """
        self.fig = plt.figure(figsize=(10, 7))
        # Adjust the subplot area to make space for widgets
        plt.subplots_adjust(bottom=0.1, top=0.85)
        # Create slider and radio button widgets
        self._create_widgets(self.fig, self.num_frames)
        
        # Setup 3D axis for visualization
        self.ax_3d = self.fig.add_subplot(111, projection='3d')
        self._setup_axes(self.ax_3d)
    
    def initialize_view_parameters(self):
        """
        Set the default view angle and link the widgets to their callbacks.
        """
        # Set the default view based on the 'Default' entry in the views dictionary
        default_view = self.views.get('Default', (-45, 35))
        self.current_azim, self.current_elev = default_view
        self.current_roll = 0

        # Link slider and radio button events to their respective update methods
        self.slider.on_changed(self.update)
        self.view_radio.on_clicked(self._set_view)
        
        # Initialize with the first frame
        self.update(0)

    def _create_widgets(self, fig, num_frames):
        """
        Create interactive widgets: a slider for frame control and radio buttons for view selection.
        
        Args:
            fig: Matplotlib figure object.
            num_frames: Total number of frames available.
        """
        # Create a slider for frame navigation
        ax_slider = fig.add_axes([0.1, 0.01, 0.8, 0.03])
        self.slider = Slider(ax_slider, 'Frame', 0, num_frames-1, valinit=0, valstep=1)
        
        # Create radio buttons to switch between different preset views
        ax_radio = fig.add_axes([0.04, 0.8, 0.2, 0.1])
        self.view_radio = RadioButtons(ax_radio, ['Default', 'Front', 'Side', 'Top'])
        
        
        # Create a play button
        ax_play = fig.add_axes([0.04, 0.1, 0.1, 0.03])
        self.play_button = Button(ax_play, "Play")
        self.play_button.on_clicked(self.toggle_play)
        
    def _initialize_visual_elements(self, ax):
        """
        Initialize the scatter and line elements on the 3D axis.
        
        Args:
            ax: Matplotlib 3D axis.
            
        Returns:
            scatter: The scatter plot object for pose landmarks.
            lines: List of line objects for the skeletal connections.
        """
        # Create an empty scatter plot for the landmarks
        scatter = ax.plot([], [], [], marker='o', markersize=5, color=self.dot_color, linestyle='')[0]
        # Create line objects for each skeletal connection defined in _CONNECTIONS
        lines = [ax.plot([], [], [], color=self.color, linestyle='-')[0] for _ in self._CONNECTIONS]
        return scatter, lines
    
    def _setup_axes(self, ax=None):
        """
        Set axis limits and labels for the 3D plot.
        
        Args:
            ax: Matplotlib 3D axis (if None, use self.ax_3d).
        """
        if ax is None:
            ax = self.ax_3d
        # Set the limits for the axes
        ax.set_xlim(self.x_axes)
        ax.set_zlim(self.z_axes)
        ax.set_ylim(self.y_axes)
        # Label the axes appropriately
        ax.set_xlabel('X')
        ax.set_ylabel('Depth (Z)')
        ax.set_zlabel('Height (Y)')
    
    # ------------------------------ Update Methods ------------------------------ #
    def _set_view(self, label):
        """
        Update the view angle based on the selected radio button.
        
        Args:
            label: Selected view label.
        """
        self._handle_view_change(label)
        self.fig.canvas.draw_idle()

    def _safe_extract_coords(self, landmarks):
        """
        Extracts x, y, and z coordinates from landmarks.
        If a landmark or its coordinate is None, returns np.nan for that coordinate.
        If the whole frame is None, returns lists of NaNs based on the expected number of landmarks.
        """
        if landmarks is None:
            if self.skeleton_data and self.skeleton_data[0] is not None:
                num_landmarks = len(self.skeleton_data[0])
            else:
                num_landmarks = 0
            return [np.nan] * num_landmarks, [np.nan] * num_landmarks, [np.nan] * num_landmarks
        
        x = [lm.x if lm is not None and lm.x is not None else np.nan for lm in landmarks]
        y = [lm.y if lm is not None and lm.y is not None else np.nan for lm in landmarks]
        z = [lm.z if lm is not None and lm.z is not None else np.nan for lm in landmarks]
        return x, y, z

    def update(self, val):
        """
        Update the visualization for a given frame index.
        """
        frame_idx = int(val)
        frame_points = self.skeleton_data[frame_idx]
        # Use the safe extraction method here
        x, y, z = self._safe_extract_coords(frame_points)

        # Update the scatter and line elements with new coordinates
        self._update_pose(x, y, z)
        self._update_widgets(frame_idx)
        self.set_axes(
            (self.min(x), self.max(x)),
            (self.min(y), self.max(y)),
            (self.min(z), self.max(z)),
        )
        self.fig.canvas.draw_idle()
        
    def _handle_view_change(self, label):
        """
        Change the 3D view based on a preset.
        
        Args:
            label: Name of the view preset.
        """
        # Retrieve azimuth and elevation from the preset views
        azim, elev = self.views[label]
        self.current_azim, self.current_elev = azim, elev
        self.ax_3d.view_init(elev=elev, azim=azim)

    def _update_widgets(self, frame_idx):
        """
        Update the plot title and slider text to reflect the current frame.
        
        Args:
            frame_idx: The current frame index.
        """
        self.ax_3d.set_title(f"Frame {frame_idx}")
        self.slider.valtext.set_text(str(frame_idx))
        
    def _update_pose(self, x, y, z):
        """
        Update the position of the scatter points and lines based on the new coordinates.
        
        Args:
            x: List of x-coordinates.
            y: List of y-coordinates.
            z: List of z-coordinates.
        """
        # Update scatter plot data
        self.scatter.set_data(x, z)
        self.scatter.set_3d_properties(y)
        # Update each line corresponding to skeletal connections
        for i, (start, end) in enumerate(self._CONNECTIONS):
            self.lines[i].set_data([x[start], x[end]], [z[start], z[end]])
            self.lines[i].set_3d_properties([y[start], y[end]])

    def set_axes(self, x: tuple, y: tuple, z: tuple):
        """
        Adjust the 3D axes limits to keep the aspect ratio and center the plot.
        
        Args:
            x: Tuple containing (min, max) for x-axis.
            y: Tuple containing (min, max) for y-axis.
            z: Tuple containing (min, max) for z-axis.
        """
        # Find the bound with the largest range to use as reference
        largest_bound = x
        for bound in [x, y, z]:
            if abs(bound[0] - bound[1]) > abs(largest_bound[0] - largest_bound[1]):
                largest_bound = bound

        largest_range = abs(largest_bound[0] - largest_bound[1])

        def safe_scale(arr):
            """
            Scale the axis limits to ensure a uniform range.
            
            Args:
                arr: List or tuple of two axis limits (min and max).
                
            Returns:
                Scaled limits as a numpy array.
            """
            # Convert the input to a numpy array of floats
            arr = np.array(arr, dtype=float)
            # Filter out any NaN values
            arr = arr[~np.isnan(arr)]
            
            # If there are fewer than 2 values, return a default range
            if len(arr) < 2:
                if len(arr) == 1:
                    # If only one valid value, set a small range around it.
                    return np.array([arr[0] - 0.5, arr[0] + 0.5])
                else:
                    # If no valid values, return a generic range.
                    return np.array([0, 1])
            
            # Compute the difference and the midpoint
            diff = abs(arr[1] - arr[0])
            mid_point = (arr[0] + arr[1]) / 2
            
            # If there's no range, return the original values
            if diff == 0:
                return arr
            
            # Scale the array based on the largest range (outer variable)
            return (arr - mid_point) / diff * largest_range + mid_point



        # Note: z and y are swapped because MediaPipe's coordinate system differs from Matplotlib's.
        self.x_axes = safe_scale(x)
        self.y_axes = safe_scale(z)
        self.z_axes = safe_scale(y)

        # Apply updated axis limits
        self.ax_3d.set_xlim(self.x_axes)
        self.ax_3d.set_zlim(self.z_axes)
        self.ax_3d.set_ylim(self.y_axes)
        
    # ------------------------------ Play Button Methods ------------------------------ #
    def toggle_play(self, event):
        """
        Toggle the play/pause state. When playing, a timer repeatedly increments the frame.
        Loops back to the first frame when reaching the end.
        """
        if not self.is_playing:
            self.is_playing = True
            self.play_button.label.set_text("Pause")
            if self.timer is None:
                self.timer = self.fig.canvas.new_timer(interval=10)  # interval in ms; adjust for desired speed
                self.timer.single_shot = False
                self.timer.add_callback(self.play_step)
            self.timer.start()
        else:
            self.is_playing = False
            self.play_button.label.set_text("Play")
            if self.timer is not None:
                self.timer.stop()

    def play_step(self):
        """
        Advance to the next frame; loop back to the first frame at the end.
        """
        current_frame = int(self.slider.val)
        next_frame = current_frame + 1
        if next_frame >= self.num_frames:
            next_frame = 0  # Loop back to the first frame
        self.slider.set_val(next_frame)

    # ------------------------------ Other Methods ------------------------------ #
    def save_video(self, direction: str, output_filename: str, fps: int = 30):
        """
        Save an animation of the 3D pose sequence as a video file.
        
        Args:
            direction: The view direction (e.g., 'Front', 'Side', 'Top', 'Default').
            output_filename: Filename for the saved video.
            fps: Frames per second for the video.
        """
        skeleton_data = np.array(self.get_skeleton_points())
        num_frames = len(skeleton_data)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        self._setup_axes(ax)

        # Map view direction to preset label
        view_mapping = {
            'Front':   'Front',
            'Side':    'Side',
            'Top':     'Top',
            'Default': 'Default'
        }
        view_label = view_mapping.get(direction, 'Default')
        self._handle_view_change(view_label)

        scatter, lines = self._initialize_visual_elements(ax)

        def update_frame(frame_idx):
            """
            Update function for each frame of the animation.
            
            Args:
                frame_idx: Index of the current frame.
            
            Returns:
                Updated scatter and line objects.
            """
            frame_points = skeleton_data[frame_idx]
            x, y, z = self._safe_extract_coords(frame_points)
            
            scatter.set_data(x, z)
            scatter.set_3d_properties(y)
            for i, (start, end) in enumerate(self._CONNECTIONS):
                lines[i].set_data([x[start], x[end]], [z[start], z[end]])
                lines[i].set_3d_properties([y[start], y[end]])
            ax.set_title(f"Frame {frame_idx}")
            return scatter, *lines

        # Create the animation
        anim = FuncAnimation(fig, update_frame, frames=num_frames, interval=1000/fps, blit=False)
        writer = FFMpegWriter(fps=fps)
        anim.save(output_filename, writer=writer)
        plt.close(fig)

    def print_view_angle(self, event):
        """
        Print the current view angles (azimuth and elevation) to the console.
        
        Args:
            event: The event that triggered this callback.
        """
        azim = self.ax_3d.azim
        elev = self.ax_3d.elev
        print(f"Current View Angle -> Azimuth: {azim}, Elevation: {elev}")
        
    def min(self, arr):
        if all(x is None for x in arr): 
            return 0
            
        return np.nanmin(arr) if len(arr) > 0 else 0
        

    def max(self, arr):
        if all(x is None for x in arr): 
            return 1
            
        return np.nanmax(arr) if len(arr) > 0 else 0
        

# ------------------------------------------------------------------------------
# PoseViewer3D: Single skeleton viewer with custom color and label.
# ------------------------------------------------------------------------------
class PoseViewer3D(Viewer):
    """
    Visualizes a single 3D skeleton sequence.
    
    Attributes:
        seq: An instance of PoseSequence or PoseWaveforms containing the pose data.
        color: Color used for drawing skeletal lines.
        dot_color: Color used for the pose landmarks.
        label: Label for the skeleton (used in legends when multiple viewers are combined).
    """
    def __init__(self, seq: Union[PoseSequence], color='blue', dot_color='grey', label='Skeleton'):
        super().__init__()
        self.seq = seq
        self.color = color
        self.dot_color = dot_color
        self.label = label

    def get_skeleton_points(self):
        """
        Retrieve the list of landmark points for each frame in the sequence.
        
        Returns:
            A list where each element contains the landmarks of a frame.
        """
        return [frame.get_landmarks() for frame in self.seq.get_sequence()]

    def __add__(self, other):
        """
        Overload the addition operator to combine multiple PoseViewer3D instances.
        
        Args:
            other: Another PoseViewer3D instance.
        
        Returns:
            A MultiPoseViewer3D instance containing both skeletons.
        """
        if isinstance(other, PoseViewer3D):
            return MultiPoseViewer3D([self, other])
        else:
            return NotImplemented

# ------------------------------------------------------------------------------
# MultiPoseViewer3D: Displays multiple skeletons with individual colors and a legend.
# ------------------------------------------------------------------------------
class MultiPoseViewer3D(Viewer):
    """
    Visualizes multiple 3D skeleton sequences simultaneously.
    
    Attributes:
        viewers: A list of PoseViewer3D instances.
        skeletons_data: A list containing the skeleton data for each viewer.
        num_frames: Total number of frames (minimum across all sequences).
    """
    def __init__(self, viewers):
        super().__init__()
        self.viewers = viewers
        # Get the skeleton data for each viewer
        self.skeletons_data = [viewer.get_skeleton_points() for viewer in viewers]
        # Use the minimum number of frames among all sequences to ensure synchronization
        self.num_frames = min(len(data) for data in self.skeletons_data)

    def __add__(self, other):
        """
        Overload the addition operator to add more PoseViewer3D or MultiPoseViewer3D instances.
        
        Args:
            other: A PoseViewer3D or MultiPoseViewer3D instance.
            
        Returns:
            A new MultiPoseViewer3D instance with the combined viewers.
        """
        if isinstance(other, PoseViewer3D):
            return MultiPoseViewer3D(self.viewers + [other])
        elif isinstance(other, MultiPoseViewer3D):
            return MultiPoseViewer3D(self.viewers + other.viewers)
        else:
            return NotImplemented
        
    def show(self):
        """
        Set up the plot, add a legend, initialize visual elements for each skeleton,
        and display the interactive multi-skeleton visualization.
        """
        # Create legend handles based on each viewer's color and label
        handles = [mpatches.Patch(color=viewer.color, label=viewer.label) for viewer in self.viewers]
        self.setup_plot()

        # Initialize visual elements (scatter and lines) for each viewer
        self.visuals = []  # List of tuples: (scatter, lines) for each viewer
        for viewer in self.viewers:
            scatter, lines = viewer._initialize_visual_elements(self.ax_3d)
            self.visuals.append((scatter, lines))
        
        self.ax_3d.legend(handles=handles)
        self.initialize_view_parameters()
        plt.show()
        
    def update(self, val):
        """
        Update the visualization for all skeletons at a given frame.
        
        Args:
            val: The current frame index from the slider.
        """
        frame_idx = int(val)
        all_x, all_y, all_z = [], [], []
        
        # Update each viewer's pose
        for idx, viewer in enumerate(self.viewers):
            data = self.skeletons_data[idx][frame_idx]
            # Safely extract coordinates by replacing None with np.nan
            x, y, z = self._safe_extract_coords(data)
            
            # Collect all coordinates to adjust global axis limits
            all_x.extend(x)
            all_y.extend(y)  
            all_z.extend(z) 
            
            # Update the scatter and line data for this viewer
            scatter, lines = self.visuals[idx]
            scatter.set_data(x, z)
            scatter.set_3d_properties(y)
            for i, (start, end) in enumerate(viewer._CONNECTIONS):
                lines[i].set_data([x[start], x[end]], [z[start], z[end]])
                lines[i].set_3d_properties([y[start], y[end]])
        self.ax_3d.set_title(f"Frame {frame_idx}")
        self.slider.valtext.set_text(str(frame_idx))
        self.fig.canvas.draw_idle()
        
        # Adjust the axes limits based on combined coordinates from all skeletons
        self.set_axes(
            (self.min(all_x), self.max(all_x)),
            (self.min(all_y), self.max(all_y)),
            (self.min(all_z), self.max(all_z)),
        )
