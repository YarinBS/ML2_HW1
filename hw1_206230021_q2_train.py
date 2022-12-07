import pickle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

from hw1_206230021_q1_train import NeuralNetwork

EPOCHS = 100  # Hyper-parameters
BATCH_SIZE = 128  # (constants)


def fetch_MNIST_data():
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
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=BATCH_SIZE,
                                              shuffle=False,
                                              drop_last=True)  # We drop the last batch to avoid a case where the last batch size is not 128, which causes some problems during runtime

    return train_dataset, train_loader, test_dataset, test_loader


def plot_graphs(train_values: list, test_values: list, mode: str) -> None:
    """
    Plots a graph of the CELoss/Accuracy values as a function of the number of epochs
    """
    plt.plot(list(range(1, len(train_values) + 1)), train_values)
    plt.plot(list(range(1, len(test_values) + 1)), test_values)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(f'{mode}')
    plt.title(f'{mode} convergence on the train and test sets')
    plt.savefig(f'{mode}_over_epochs.png')
    plt.show()


def main():
    # Fetching MNIST
    MNIST_train_data, MNIST_train_loader, MNIST_test_data, MNIST_test_loader = fetch_MNIST_data()

    # Initializing NN instance
    net = NeuralNetwork(input_size=28 * 28,
                        hidden_size=250,
                        output_size=2)

    train_images, train_labels = next(iter(MNIST_train_loader))  # Fetching the first 128 training samples
    train_images = train_images.view(-1, 28 * 28)
    train_labels = torch.randint(low=0, high=2, size=(128,))  # Using randint() for random label generation
    train_labels_one_hot = torch.nn.functional.one_hot(train_labels.long(),
                                                       2)  # Used to match the dimensions in the train phase

    loss = torch.nn.CrossEntropyLoss()

    loss_test = []
    loss_train = []
    accuracy_test = []
    accuracy_train = []

    for i in range(EPOCHS):
        # print(i)
        # Training the model
        net.train(train_images, train_labels_one_hot)

        # Generating predictions for the train set
        y_train_predictions = torch.argmax(net.forward(train_images), dim=1)

        # Calculating train set accuracy and saving it
        total_predictions = train_labels.size(0)
        correct_predictions = (y_train_predictions == train_labels).sum()
        accuracy_train.append(100 * (correct_predictions / total_predictions))

        # Calculating loss for train (Later - saving it)
        net_train_loss = loss(input=net.forward(train_images).float(), target=train_labels.long())

        accumulated_net_test_loss, total_batches = 0, 0
        total_test_predictions, correct_test_predictions = 0, 0
        for test_images, test_labels in MNIST_test_loader:
            total_test_predictions += test_labels.size(0)
            test_images = test_images.view(-1, 28 * 28)
            test_labels = torch.randint(low=0, high=2, size=(128,))  # Using randint() for random label generation
            # test_labels_one_hot = torch.nn.functional.one_hot(test_labels.long(), 2)  # Used to match the dimensions

            # Generating prediction for the test set
            y_test_predictions = torch.argmax(net.forward(test_images), dim=1)

            # Calculating the current accuracy
            correct_test_predictions += (y_test_predictions == test_labels).sum()

            # Calculating loss for test and appending to the curr_test_loss
            accumulated_net_test_loss += loss(input=net.forward(test_images).float(), target=test_labels.long())
            total_batches += 1

        # Saving the current accuracy
        accuracy_test.append(100 * (correct_test_predictions / total_test_predictions))

        # Averaging the test loss over all epochs
        net_test_loss = accumulated_net_test_loss / total_batches

        # Saving loss values for plotting
        loss_train.append(net_train_loss.item())
        loss_test.append(net_test_loss.item())

    # Plotting loss convergence
    plot_graphs(loss_train, loss_test, 'CE Loss')

    # Plotting accuracy convergence
    plot_graphs(accuracy_train, accuracy_test, 'Accuracy')

    # Saving the trained model and the weights
    with open("q2_model.pkl", "wb") as f:
        pickle.dump(net, f)
    torch.save({'w1': net.W1, 'b1': net.b1, 'w2': net.W2, 'b2': net.b2}, 'q2_weights.pkl')


if __name__ == '__main__':
    main()
