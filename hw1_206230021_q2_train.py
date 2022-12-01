import pickle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np
from matplotlib import pyplot as plt

from hw1_206230021_q1_train import NeuralNetwork


def fetch_MNIST_train():
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
                                               batch_size=128,
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=128,
                                              shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader


# Fetching MNIST
MNIST_train_data, MNIST_train_loader, MNIST_test_data, MNIST_test_loader = fetch_MNIST_train()

# Initializing NN instance
net = NeuralNetwork(input_size=28 * 28,
                    hidden_size=250,
                    output_size=10)

train_images, train_labels = next(iter(MNIST_train_loader))  # Fetching the first 128 training samples
train_labels = torch.nn.functional.one_hot(train_labels.long(), 10)  # Random Bernoulli labels to the training samples
train_images = train_images.view(-1, 28 * 28)
net.train(train_images, train_labels)

loss = torch.nn.CrossEntropyLoss()

# -----------------------

loss_test = []
loss_train = []
accuracy_test = []
accuracy_train = []
for epoch in range(1001):
    net.train(train_images, train_labels)
    if epoch % 10 == 0:
        com_loss = 0
        predicted = net.forward(train_images.view(-1, 28 * 28))
        o = torch.argmax(predicted, dim=1)
        com_loss = loss(input=predicted.float(), target=train_labels.long())

        loss_train.append(com_loss)
        print('train Loss with ' + str(epoch) + ' epochs is - ' + str(com_loss))
        # accuracy train
        correct = 0
        correct = (o == labels).sum()
        accuracy_train.append(100 * correct / 128)

        com_loss = 0
        total = 0
        n = 0
        correct = 0
        for images_test, labels_test in MNIST_test_loader:
            labels_test = torch.randint(0, 2, (labels_test.shape[0],))
            total += 1
            n += labels_test.size(0)
            predicted = net.forward(images_test.view(-1, 28 * 28))
            o = torch.argmax(predicted, dim=1)
            com_loss += loss(predicted.float(), labels_test.long())
            correct += (o == labels_test).sum()
        loss_test.append(com_loss / total)
        accuracy_test.append(100 * correct / n)

        print('Average test Loss with ' + str(epoch) + ' epochs is - ' + str(com_loss / total))

epochs = [i for i in range(1001) if i % 10 == 0]

plt.plot(epochs, loss_test, label="test loss")
plt.plot(epochs, loss_train, label="one batch loss")
plt.title("loss as a function of epochs")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.plot(epochs, accuracy_test, label="test accuracy")
plt.plot(epochs, accuracy_train, label="accuracy_train")
plt.title("accuracy as a function of epochs")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()
