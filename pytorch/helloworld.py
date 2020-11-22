# Implements the following 2 layer NN:
# Input=0.8, Bias=[-0.14, -0.11], Weight=[1.58, 2.45], Activation=Sigmoid
# y = sigmoid( -0.11 + 2.45 * sigmoid( -0.14 + 1.58 * 0.8 ) )
#         L1          L2
# [0.8] - O -   z   - O - y
#       Bl1=-0.14     Bl2=-0.11
#       Wl1=1.58      Wl2=2.45

import torch
from torch import nn
import pudb

# Naive implementation
# import pudb; pudb.set_trace()

x = torch.tensor([0.8])
raw_activation = (x * 1.58) - 0.14
print("L1 raw activation: {}".format(raw_activation))
z = torch.sigmoid(raw_activation)
print("L1 o/p z: {}".format(z))
y = torch.sigmoid((z * 2.45) - 0.11)
print("L2 o/p y: {}".format(y))


# Create a single layer feed forward network with 1 input and 1 output.
# The weights and biases are auto initialized by pytorch.
l1 = nn.Linear(1, 1)

# Replace the weight and bias.
# Note: nn.Linear expects its weight and bias fields to be Parameters.
# Parameters don't help unless used in the larger context of a nn.Module.
# When used in such a context, these fields will appear in model attributes.

# The bias parameter is a 1d tensor because each Neuron only has 1 bias.
l1.bias = torch.nn.Parameter(torch.tensor([-0.14]))
# The weight needs to be an array of arrays because the Neuron can have any
# number of inputs, and each of those require a weight vector.
l1.weight = torch.nn.Parameter(torch.tensor([[1.58]]))

nn_raw_act = l1(x)
print("nn L1 raw activation: {}".format(nn_raw_act))
nn_z = torch.sigmoid(nn_raw_act)
print("nn L1 o/p z: {}".format(nn_z))
nn_y = torch.sigmoid((nn_z * 2.45) - 0.11)
print("nn L2 o/p y: {}".format(nn_y))

