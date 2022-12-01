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


# --- Utility functions ---
def relu(x):
    return np.maximum(x, 0)


def reluPrime(x):
    # derivative of relu
    # x: relu output
    return (x > 0) * 1


def softmax(x):
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    return x_exp / x_exp_sum


# --- Neural Network class ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Parameters
        self.inputLayerSize = input_size
        self.hidden1Size = hidden_size
        self.outputLayerSize = output_size

        # Weights
        self.W1 = torch.randn(self.inputLayerSize, self.hidden1Size)
        self.b1 = torch.zeros(self.hidden1Size)

        self.W2 = torch.randn(self.hidden1Size, self.outputLayerSize)
        self.b2 = torch.zeros(self.outputLayerSize)

    def forward(self, X):
        # Input to hidden
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = relu(self.z1)

        # Hidden to output
        self.z2 = torch.matmul(self.h, self.W2) + self.b2

        return softmax(self.z2)

    def backward(self, X, y, y_hat, lr=LEARNING_RATE):
        batch_size = y.size(0)
        dl_dz2 = (1 / batch_size) * (y_hat - y)

        dl_dh = torch.matmul(dl_dz2, torch.t(self.W2))
        dl_dz1 = dl_dh * reluPrime(self.h)

        # Update weights
        self.W1 -= lr * torch.matmul(torch.t(X), dl_dz1)
        self.b1 -= lr * torch.matmul(torch.t(dl_dz1), torch.ones(batch_size))
        self.W2 -= lr * torch.matmul(torch.t(self.h), dl_dz2)
        self.b2 -= lr * torch.matmul(torch.t(dl_dz2), torch.ones(batch_size))

    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)


# --- Neural Network prediction ---

def nn_predict(net: NeuralNetwork, loader):
    total_predictions, correct_predictions = 0, 0
    for test_images, test_labels in loader:
        total_predictions += test_labels.size(0)
        predictions = net.forward(test_images.view(-1, 28 * 28))
        argmax_class = torch.argmax(predictions, dim=1)
        correct_predictions += (argmax_class == test_labels).sum()

    return torch.round(torch.tensor(100 * correct_predictions / total_predictions), decimals=2)


# --- Convergence plot ---

def plot_convergence(data: list, mode: str):
    plt.plot(data)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy over time on the {mode} set')
    plt.show()

# --- Fetching MNIST data ---

def fetch_MNIST():
    # Transform the image data into Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transform,
                                  download=True)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=BATCH_SIZE,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


def main():
    MNIST_train_data, MNIST_train_loader, MNIST_test_data, MNIST_test_loader = fetch_MNIST()

    # Instantiating a Neural Network
    nn = NeuralNetwork(input_size=28 * 28,
                       hidden_size=250,
                       output_size=10)

    # Training the NN
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}...")
        for i, (train_images, train_labels) in enumerate(MNIST_train_loader):
            train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)
            train_images = train_images.view(-1, 28 * 28)
            nn.train(train_images, train_labels)

    # Saving the model
    torch.save({'w1': nn.W1, 'b1': nn.b1,'w2': nn.W2, 'b2': nn.b2}, 'nn_weights.pkl')

    # Testing the NN on the train set
    train_accuracy = nn_predict(nn, MNIST_train_loader)
    print(f"Accuracy on the train set: {train_accuracy}%")

    # Testing the NN on the test set
    test_accuracy = nn_predict(nn, MNIST_test_loader)
    print(f"Accuracy on the test set: {test_accuracy}%")


if __name__ == '__main__':
    main()
