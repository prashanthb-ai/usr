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

model = Net()
for i in range(NUM_EPOCHS):
    for batch in enumerate(data_loader):
        i, (images, expected) = batch
        predicted = model(images)
        # for j in range(len(predicted)):
        #    print("Got: {}, Expected: {}".format(
        #        torch.argmax(predicted[j]), expected[j]))

    # TODO: Compute MSE loss
    # loss =


