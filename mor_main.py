import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets

# Hyper Parameters
num_epochs = 50
batch_size = 64
learning_rate = 0.05
n_classes = 10

# Image Preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))
                                ])
# MNIST Dataset
train_dataset = dsets.MNIST(root='./data/',
                            train=True,
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data/',
                           train=False,
                           transform=transform,
                           download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


def relu(Z):
    return np.maximum(Z, 0)


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def deriv_relu(Z):
    return (Z > 0) * 1


def softmax(Z):
    f = torch.exp(Z - torch.max(Z))
    return f / torch.sum(f)


def softmax_torch(x):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


def cross_entropy(activations, labels):
    return - torch.log(activations[range(labels.shape[0]), labels]).mean()


class Neural_Network:
    def __init__(self, input_size=28 * 28, output_size=10, hidden_size=250):
        # parameters
        self.inputSize = input_size
        self.outputSize = output_size
        self.hiddenSize = hidden_size

        # weights
        self.W1 = torch.randn(self.inputSize, self.hiddenSize)
        self.b1 = torch.zeros(self.hiddenSize)

        self.W2 = torch.randn(self.hiddenSize, self.outputSize)
        self.b2 = torch.zeros(self.outputSize)

    def forward(self, X):
        self.z1 = torch.matmul(X, self.W1) + self.b1
        self.h = relu(self.z1)
        self.z2 = torch.matmul(self.h, self.W2) + self.b2
        return softmax_torch(self.z2)

    def backward(self, X, y, y_hat, lr=0.05):
        batch_size = y.size(0)

        ce_grad = (1 / batch_size) * (y_hat - y)

        total_grad = torch.matmul(ce_grad, torch.t(self.W2))
        self.W2 -= lr * torch.matmul(torch.t(self.h), ce_grad)
        self.b2 -= lr * torch.matmul(torch.t(ce_grad), torch.ones(batch_size))

        relu_grad = total_grad * deriv_relu(self.z1)
        self.W1 -= lr * torch.matmul(torch.t(X), relu_grad)
        self.b1 -= lr * torch.matmul(torch.t(relu_grad), torch.ones(batch_size))

    def train(self, X, y):
        # forward + backward pass for training
        o = self.forward(X)
        # o = torch.argmax(o,dim=1)
        # o = torch.nn.functional.one_hot(o.long(),num_classes=10)
        self.backward(X, y, o)


def mor_main():
    output = Neural_Network()
    print("NN initialized")
    for epoch in range(num_epochs):
        print(f"epoch {epoch}...")
        for i, (images, labels) in enumerate(train_loader):
            labels = torch.nn.functional.one_hot(labels.long(), 10)
            images = images.view(-1, 28 * 28)
            output.train(images, labels)

    print("Done epoching")
    correct = 0
    total = 0

    for images_, labels_ in test_loader:
        # print("Iterating...")
        total += labels_.size(0)
        predicted = output.forward(images_.view(-1, 28 * 28))
        o = torch.argmax(predicted, dim=1)
        correct += (o == labels_).sum()

    print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))
