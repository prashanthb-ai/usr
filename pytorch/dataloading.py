import torch
from torchvision import transforms
from torchvision import datasets

# Batch size for image loading
BATCH_SIZE = 10

# The path to retrieve and store datasets from.
path = "/home/beeps/rtmp/datasets"

# Transformation applied to all the data in the dataset.
# This just transforms intput data (PIL images in this case) into a tensor.
transformations = transforms.Compose([transforms.ToTensor()])

# get_dataset invokes pytorch's datasets library to fetch the MNIST dataset.
# @root: Where to store/fetch the data set from
# @download: download dataset into root?
# @train: download/fetch test or train dataset?
# @transform: transform the given data using this transform
def get_dataset(root=path, train=True, transform=transformations,
        download=True):
    return datasets.MNIST(root=root, train=train, transform=transform,
            download=download)

# get_loader returns a dataloader.
# In pytorch a:
#   DataLoader: invokes DataSet in batches, shuffles
#   DataSet: Manages data/labels/iteration
#   Load: simply loads the raw data/labels from the filesystem.
#
# @batch_size: the batch size of images to train on before updating params.
#              This is also the #images SGD averages over.
# @shuffle: shuffle the input data before returning it in batches?
# @dataset: initial dataset to pull batches from.
def get_loader(dataset, batch_size=BATCH_SIZE, shuffle=True):
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size,
            shuffle=shuffle)


# get_train_loader returns the loader for the training dataset.
def get_train_loader():
    train_dataset = get_dataset()
    train_loader = get_loader(train_dataset)
    return train_loader


# get_test_loader returns a loader for the test dataset.
def get_test_loader():
    test_dataset = get_dataset(train=False)
    test_loader = get_loader(test_dataset, shuffle=False)
    return test_loader


train = get_dataset()
print("train dataset length {}".format(len(train)))
test = get_dataset(train=False)
print("test dataset length {}".format(len(test)))

# Data loader API is such that when you iterate over it, you get back 2 sets
# of "images". One of the actual image, and one of the int representation.
data_loader = get_train_loader()

# images is the actual PIL image.
# expected_results are the actual digit int representatin of each PIL image.
# The data loader returns BATCH_SIZE # of such images.
images, expected_results = next(iter(data_loader))

# Classes are mapped to indices in the output via data_loader.dataset.classes
# This shows an array, where the index corresponds to the index of the output
# neuron configured to trigger on activation, and the value is the actual label
# value to use for that classification.
print(data_loader.dataset.classes)
