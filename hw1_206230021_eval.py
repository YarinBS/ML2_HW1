import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

# --- Hyper-parameters (constants) ---

EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.05
CLASSES = 10


# --- Fetching MNIST test ---

def fetch_MNIST_test():
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform,
                                  download=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return test_dataset, test_loader


def evaluate_hw1():
    """
    This function:
    1. loads the MNIST test set
    2. loads the trained NN
    3. generates predictions on the test set
    4. returns the accuracy
    :return:
    """

    MNIST_test_data, MNIST_test_loader = fetch_MNIST_test()
    nn = torch.load('nn_weights.pkl')
