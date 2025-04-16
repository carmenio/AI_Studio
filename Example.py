# Set The Parent Directory - to main.py folder
import os, sys; [(sys.path.append(d), print(f'Added {d} to system path')) for d in (os.path.abspath(os.path.join(os.getcwd(), *([".."] * i))) for i in range(len(os.getcwd().split(os.sep)))) if os.path.isfile(os.path.join(d, 'main.py'))]


from Model.Pose.Sequence.PoseFrame import PoseFrame, Landmark
from Model.Pose.Sequence.PoseSequence import PoseSequence
from Model.Pose.View.PersonViewer3D import PoseViewer3D

pose_seq = PoseSequence.load('filepath')

for frame in pose_seq:
    frame: PoseFrame
    for landmark in frame:
        landmark: Landmark
        print(landmark)
        # print(landmark.x, landmark.y, landmark.z)
        
viewer = PoseViewer3D(pose_seq)
viewer.show()