import pandas as pd
import numpy as np

landmarks_frame = pd.read_csv(
        '/home/beeps/rtmp/datasets/faces/face_landmarks.csv')

print (np.reshape(np.asarray(landmarks_frame.iloc[0, 1:]), (-1, 2)).shape)
# TODO: Plot landmarks over jpgs


