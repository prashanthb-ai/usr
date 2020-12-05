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

import pudb; pudb.set_trace()
# Plotting boilerplat, ion activates "interactive mode" so we can keep adding
# commands to the origin plot (eg: exis, dots for the landscape etc).
plt.ion()
image = io.imread(os.path.join(data_dir, image_name), as_gray=False)
plt.imshow(image)

# image_landmarks is an array of array coordinates, eg:
# [[x, y], [x, y]...]
# We want a scatter plot with [x1, x2..] and [y1, y2...]
plt.scatter(
        image_landmarks[:,0], # x coords of the scatter plot marker
        image_landmarks[:,1], # y corrds of the scatter plot marker
        s=10, # Size of the scatter plot marker
        marker=".", # Shape of the scatter plot marker
        c='r' # Color of the scatter plot markre
)
plt.pause(1)

# TODO: implement the torch.utils.data.Dataset class
# Transform each item in the set to:
#   {image: <image>, landmarks: <landmarks for image>}
# via __getitem__
