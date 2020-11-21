# Implements the following 2 layer NN
#   Input=0.8, Bias=[-0.14, -0.11], Weight=[1.58, 2.45], Activation=Sigmoid
#   y = sigmoid( -0.11 + 2.45 * sigmoid( -0.14 + 1.58 * 0.8 ) )

import torch

x = torch.tensor([0.8])
print(x)
