# Implements the following 2 layer NN:
# Input=0.8, Bias=[-0.14, -0.11], Weight=[1.58, 2.45], Activation=Sigmoid
# y = sigmoid( -0.11 + 2.45 * sigmoid( -0.14 + 1.58 * 0.8 ) )
#         L1          L2
# [0.8] - O -   z   - O - y
#       Bl1=-0.14     Bl2=-0.11
#       Wl1=1.58      Wl2=2.45

import torch
from torch import nn
from torch import optim

# Naive implementation
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


# NN defines a Neural Network.
#
# The input weights and bias vectors are expected to be of the same length.
# The number of layers is taken as the size of these vectors, since each layer
# is a 1 input 1 output Neuron.
#
# @weights: a list of weights.
# @bias: a list of bias values.
class NN(nn.Module):


    def __init__(self, weights, bias):
        super(NN, self).__init__()
        layers = []
        for i in range(len(weights)):
            l = nn.Linear(1, 1)
            l.weight = torch.nn.Parameter(torch.tensor([[weights[i]]]))
            l.bias = torch.nn.Parameter(torch.tensor([bias[i]]))
            layers.append(l)
        self.layers = nn.ModuleList(layers)


    # Forward takes a tensor 'x' and feeds it forward through the network.
    def forward(self, x):
        for i in range(len(self.layers)):
            x = torch.sigmoid(self.layers[i](x))
            print("NN L{} o/p: {}".format(i, x))
        return x


model = NN([1.58, 2.45], [-0.14, -0.11])
output = model(x)
print("NN Output: {}".format(output))

# Backpropogation
# The dependency graph for this NN:
#
#         L1               L2
# [0.8] - O -   a(l-1)   - O - zl - sigmoid - al(l=2) - predicted - c - MSE = 1/2(al - y)^2
#         Bl1=-0.14        Bl2=-0.11                                |
#         Wl1=1.58         Wl2=2.45                                 y = real output (1.0)
#
# So we differentiate dc/dal = (al - y)
# Then we substitute al = sigmoid(zl) which means dal/dzl = sigmoid'(zl)
# And we apply the chain rule to compute:
# dc/dbl2 = dc/dal2 * dal2/dzl2 * dzl2/dbl2
# dc/dwl2 = dc/dal2 * dal2/dzl2 * dzl2/dwl2
# dc/da(l-1) = dc/dal2 * dal2/dzl2 * dzl2/da(l-1)
#
# Substituting the following values:
#
# dzl2/dbl2 = 1 (zl2 = bl2 + C)
# dzl2/dwl2 =  a(l-1) (zl2 = C + wl2 * a(l-1))
# dal/dzl2 = sigmoid'(z) = sigmoid(z)(1 - sigmoid(z)) where sigmoid(z) = 1/(1+e^-z)
# dc/dal2 = (al-y)
#
# And update the weight and bias of l2:
# updated_wl2 = wl2 + dc/dwl2 * step
# updated_bl2 = bl2 + dc/dbl2 * step
#
# Similarly we can compute the updated values of wl1 and bl2:
# dc/dbl1 = dc/dal1 * dal1/dzl1 * dzl1/dbl1
# dc/dwl1 = dc/dal1 * dal1/dzl1 * dzl1/dwl1
#
# And update the weight and dias of l1:
# updated_wl1 = wl1 + dc/dwl1 * step
# updated_bl1 = bl1 + dc/dbl1 * step

y = torch.tensor([1.0])
cost = nn.MSELoss()

# Compute MSE for output
loss = cost(output, y)

# Zero out the gradient buffers for all parameters
model.zero_grad()

# Compute dloss/d(parameter) for all parameters which require gradient
loss.backward()

# Define and run the Gradient Descent optimizer
# This will apply the function: parameter = -lr * parameter.grad
optimizer = optim.SGD(model.parameters(), lr=0.1)
optimizer.step()

for i in range(len(model.layers)):
    print("NN: Updated bias for l{} {}".format(i, model.layers[i].bias.item()))
    print("NN: Updated weight for l{} {}".format(i, model.layers[i].weight.item()))

output = model(x)
print("NN: optimized output {}".format(output))

