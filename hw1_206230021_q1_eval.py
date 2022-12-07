import pickle
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from hw1_206230021_q1_train import nn_predict, NeuralNetwork

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

    # Fetching MNIST test data
    MNIST_test_data, MNIST_test_loader = fetch_MNIST_test()

    # Loading the pretrained model
    nn = pickle.load(open("q1_model.pkl", "rb"))

    # Testing the NN
    test_accuracy = nn_predict(nn, MNIST_test_loader)
    print(f"Accuracy on the test set: {test_accuracy}%")
    return test_accuracy


def main():
    evaluate_hw1()


if __name__ == '__main__':
    main()
