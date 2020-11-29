import torch
import dataloader

from torch import nn

# Size of input images
INPUT_SIZE = 28 * 28

# Number of output classes
OUTPUT_SIZE = 10

# Number of training iterations
NUM_EPOCHS = 30

# Increments of modification of wts/bias
LEARNING_RATE = 3.0

# Net is a dense NN for image classification
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 100)
        self.output_layer = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE)
        x = torch.sigmoid(self.hidden_layer(x))
        return torch.sigmoid(self.output_layer(x))


data_loader = dataloader.get_train_loader()

# Invokes the provided loss function after transforming the inputs.
def _loss(pred, expected, loss_fn):
    normalized_expected = []
    # The expected values are ints, while the predicted values are arrays of
    # probabilities. Transform the former to be more like the latter, i.e
    # expected = [1,2] -> [[1.0, 0.0, ... len(pred)], [0.0, 1.0, ..len(pred)]]
    for i in expected:
        ne = [0.0 for _ in pred]
        ne[i] = 1.0
        normalized_expected.append(ne)
    normalized_expected = torch.tensor(normalized_expected)
    return loss_fn(pred, normalized_expected)


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for i in range(NUM_EPOCHS):
    for batch in enumerate(data_loader):

        # Get one data set from the loader
        i, (images, expected) = batch

        # Run the forward pass of the model to produce an array (one for each
        # expected value/input image) of arrays (one for each output class) of
        # probabilities.
        predicted = model(images)

        # Compute the MSE loss.
        loss = _loss(predicted, expected, loss_fn)

        # TODO: optimize and call loss.backwards()
