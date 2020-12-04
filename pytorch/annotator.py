import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import os

data_dir = "/home/beeps/rtmp/datasets/faces"
landmarks_frame = pd.read_csv(os.path.join(data_dir, "face_landmarks.csv"))

# iloc pulls out a single row, leaving out the first column (which is the name
# of the image). Numpy reshape converts a 1d array into a matrix of the given
# shape. A -1 as the first argument means use as many rows as needed but only 2
# columns / row.
image_index = 65
image_name = landmarks_frame.iloc[image_index, 0]
image_landmarks = np.reshape(np.asarray(landmarks_frame.iloc[image_index, 1:]), (-1, 2))
print("image_name {}".format(image_name))
print("image_landmarks shape {}".format(image_landmarks.shape))

# Plotting boilerplat, ion activates "interactive mode".
# TODO: What's the standard way to plot an image?
# plt.ion()
image = io.imread(os.path.join(data_dir, image_name), image_name)
plt.imshow(image)
plt.pause(1)

# TODO: Plot landmarks over jpgs using plt.scatter
# TODO: Figure out discoloration
#plt.figure()
#plt.show()

