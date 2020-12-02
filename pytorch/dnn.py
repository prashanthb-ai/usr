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


# Tests the given model with dat from the given loader.
# Prints out ratio of correct predictions / total values.
def test_model(model, loader, epoch):
    # Prepare the model for evaluation by turning off certain preconfigured
    # layers that shouldn't be activated during evaluation.
    model.eval()
    # Tell pytorch to stop tracking gradients. This, along with model.eval,
    # allows us to reuse the same test dataset repeatedly because no state
    # is remembered.
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in loader:
            images, expected = batch
            output = model(images)
            predicted = torch.argmax(output, dim=1)
            correct += (predicted == expected).sum()
            total += len(expected)
        print("Epoch {}: correct: {}, total: {}".format(
            epoch, correct, total))


model = Net()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()

for epoch in range(NUM_EPOCHS):
    # Activate stateful parts of the model for training
    model.train()
    for batch in enumerate(data_loader):
        # Get one data set from the loader
        _, (images, expected) = batch

        # Run the forward pass of the model to produce an array (one for each
        # expected value/input image) of arrays (one for each output class) of
        # probabilities.
        #
        # Pytorch builds a graph of all this models parameters for subsequent
        # gradient computations via loss.backward().
        predicted = model(images)

        # Zero out pytorch's runtime heap gradient buffers.
        optimizer.zero_grad()

        # Compute the MSE loss.
        # This step actually computes the loss and differentiates the loss
        # function wrt all parameters marked as requiring a gradient
        # (requires_gradient=True).
        loss = _loss(predicted, expected, loss_fn)

        # Compute gradients via back propogation.
        # This operates on the intermediate NN layer of the model stored by
        # pytorch during the forward pass of the model.
        loss.backward()

        # Apply the computed gradients to weight/bias update.
        # NB: The loss and optimize functions are connected to the model via
        # pytorch.
        optimizer.step()

    test_model(model, dataloader.get_test_loader(), epoch)

