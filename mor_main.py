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


# def one_hot(y):
#     max_idx = torch.argmax(y, 0, keepdim=True)
#     one_hot = torch.FloatTensor(y.shape)
#     one_hot.zero_()
#     one_hot.scatter_(0, max_idx, 1)
#     return one_hot

def relu(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def sigmoid(s):
    return 1 / (1 + torch.exp(-s))


def deriv_relu(Z):
    return (Z > 0) * 1


# def deriv_softmax(Z):
#     dZ = np.exp(Z) / sum(np.exp(Z)) * (1. - np.exp(Z) / sum(np.exp(Z)))
#     return dZ

def softmax(Z):
    f = torch.exp(Z - torch.max(Z))
    return f / torch.sum(f)

    # numerator = torch.exp(Z)
    # denominator = sum(torch.exp(Z))
    # A = torch.exp(Z) / sum(torch.exp(Z))
    # return A


def softmax_torch(x):  # Assuming x has atleast 2 dimensions
    maxes = torch.max(x, 1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)
    x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    probs = x_exp / x_exp_sum
    return probs


#
# class CrossEntropyLossManual:
#     """
#     y0 is the vector with shape (batch_size,C)
#     x shape is the same (batch_size), whose entries are integers from 0 to C-1
#     """
#
#     def __init__(self, ignore_index=-100) -> None:
#         self.ignore_index = ignore_index
#
#     def __call__(self, y0, x):
#         loss = 0.
#         n_batch, n_class = y0.shape
#         # print(n_class)
#         for y1, x1 in zip(y0, x):
#             class_index = int(x1.item())
#             if class_index == self.ignore_index:
#                 n_batch -= 1
#                 continue
#             loss = loss + torch.log(torch.exp(y1[class_index]) / (torch.exp(y1).sum()))
#         loss = - loss / n_batch
#         return loss

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
