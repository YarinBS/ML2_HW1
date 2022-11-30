import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import math


# --- Utility functions ---
def relu(x):
    return torch.tensor(max(0, x))


def reluPrime(x):
    # derivative of relu
    # x: relu output
    try:
        x[x <= 0] = 0
        x[x > 0] = 1
        return torch.tensor(x)
    except Exception as e:
        y = (x > 0) * 1
        return torch.tensor(y)


def lrelu(x):
    return torch.tensor(max(0.1 * x, x))


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def sigmoidPrime(s):
    # derivative of sigmoid
    # s: sigmoid output
    return torch.tensor(s * (1 - s))


def tanh(t):
    return torch.div(torch.exp(t) - torch.exp(-t), torch.exp(t) + torch.exp(-t))


def tanhPrime(t):
    # derivative of tanh
    # t: tanh output
    return torch.tensor(1 - t * t)


# ----------------------

# --- Neural Network class ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size):
        # Parameters
        self.inputLayerSize = input_size
        self.hidden1Size = hidden_1_size  # Number of neurons in the first hidden layer
        self.hidden2Size = hidden_2_size  # Number of neurons in the second hidden layer
        self.hidden3Size = hidden_3_size  # Number of neurons in the third hidden layer
        self.outputLayerSize = output_size

        # Weights
        self.W1 = torch.randn(self.inputLayerSize, self.hidden1Size)
        self.b1 = torch.zeros(self.hidden1Size)

        self.W2 = torch.randn(self.hidden1Size, self.hidden2Size)
        self.b2 = torch.zeros(self.hidden2Size)

        self.W3 = torch.randn(self.hidden2Size, self.hidden3Size)
        self.b3 = torch.zeros(self.hidden3Size)

        self.W4 = torch.randn(self.hidden3Size, self.outputLayerSize)
        self.b4 = torch.zeros(self.outputLayerSize)

    def forward(self, X):
        # Input to 1st hidden
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h1 = relu(self.z1)

        # 1st hidden to 2nd hidden
        self.z2 = torch.matmul(self.h1, self.W2) + self.b2
        self.h2 = relu(self.z2)

        # 2nd hidden to 3rd hidden
        self.z3 = torch.matmul(self.h2, self.W3) + self.b3
        self.h3 = relu(self.z3)

        # 3rd hidden to output
        self.z4 = torch.matmul(self.h3, self.W4) + self.b4

        return sigmoid(self.z4)

    def backward(self, X, y, y_hat, lr=.1):
        batch_size = y.size(0)
        dl_dz2 = (1 / batch_size) * (y_hat - y)

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * tanhPrime(self.h)

        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        self.backward(X, y, o)


# ------------------

# --- Fetching MNIST data ---

def fetch_MNIST():
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.4914, 0.4822, 0.4465),
        #                      (0.247, 0.2434, 0.2615)),
    ])

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform,
                                  download=True)

    batch_size = 100

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


# ---------------------------


def main():
    MNIST_train_data, MNIST_train_loader, MNIST_test_data, MNIST_test_loader = fetch_MNIST()

    # Instantiating a Neural Network class
    nn = NeuralNetwork(input_size=28 * 28,
                       hidden_1_size=500,
                       hidden_2_size=250,
                       hidden_3_size=100,
                       output_size=10)

    nn.forward(MNIST_train_data)


if __name__ == '__main__':
    main()
