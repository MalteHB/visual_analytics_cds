
import argparse

from utils.utils import fetch_mnist
from utils.neuralnetwork_malte import NeuralNetworkMalte


import numpy as np

# Import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


def main(args):

    print("Initiating some awesome neural network classification!")

    # Importing arguments from the arguments parser

    random_state = args.rs

    test_size = args.ts

    scaling = args.s

    minmax = args.mm

    epochs = args.e

    early_stopping = args.es

    nn_mnist = NeuralNetworkMNIST()

    X_train, X_test, y_train, y_test = nn_mnist.split_and_preprocess_data(random_state=random_state,
                                                                          test_size=test_size,
                                                                          scaling=scaling,
                                                                          minmax=minmax)

    nn_model = nn_mnist.train(X_train, y_train, epochs=epochs, early_stopping=early_stopping)

    nn_mnist.print_eval_metrics(nn_model, X_test, y_test)

    print("DONE! Have a nice day. :-)")


class NeuralNetworkMNIST:

    def __init__(self):

        self.X, self.y = fetch_mnist()

    def split_and_preprocess_data(self, random_state=1, test_size=0.2, scaling=False, minmax=True):
        """Splits the data into a train/test-split

        Args:
            random_state (int, optional): The random state. Defaults to 9.
            train_size (int, optional): Size of the training data. Defaults to 7500.
            test_size (int, optional): Size of the test data. Defaults to 2500.

        Returns:
            X_train, X_test, y_train, y_test: Train/test-split of the data.
        """

        self.X = np.array(self.X)
        self.y = np.array(self.y)

        if scaling:

            X_norm = self.X / 255.0

        elif minmax:

            X_norm = (self.X - self.X.min()) / (self.X.max() - self.X.min())

        else:

            X_norm = self.X

        X_train, X_test, y_train, y_test = train_test_split(X_norm,
                                                            self.y,
                                                            random_state=random_state,
                                                            test_size=test_size)

        # convert labels from integers to vectors
        y_train = LabelBinarizer().fit_transform(y_train)
        y_test = LabelBinarizer().fit_transform(y_test)

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, epochs=100, early_stopping=True):

        # train network
        nn_model = NeuralNetworkMalte([X_train.shape[1], 32, 16, 10])

        nn_model.fit(X_train, y_train, epochs=epochs, early_stopping=early_stopping)

        return nn_model

    def print_eval_metrics(self, nn_model, X_test, y_test):
        """Prints the evaluation metrics to the terminal.

        Args:
            nn_model (sklearnModel): [description]
            X_test ([type]): [description]
            y_test ([type]): [description]
        """

        # evaluate network

        y_pred = nn_model.predict(X_test)

        y_pred = y_pred.argmax(axis=1)

        print(metrics.classification_report(y_test.argmax(axis=1), y_pred))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--rs',
                        metavar="Random State",
                        type=int,
                        help='Random State of the logistic regression model.',
                        required=False,
                        default=1)

    parser.add_argument('--ts',
                        metavar="Test Size",
                        type=int,
                        help='The test size of the test data.',
                        required=False,
                        default=2500)

    parser.add_argument('--s',
                        metavar="Scaling",
                        type=bool,
                        help='Whether to scale the data of not to.',
                        required=False,
                        default=False)

    parser.add_argument('--mm',
                        metavar="MinMax",
                        type=bool,
                        help='Whether to MinMax normalize the data of not to.',
                        required=False,
                        default=True)

    parser.add_argument('--e',
                        metavar="Epochs",
                        type=int,
                        help='Number of epochs for the neural network training.',
                        required=False,
                        default=100)

    parser.add_argument('--es',
                        metavar="Early Stopping",
                        type=bool,
                        help='Whether to stop the training before overfitting the data.',
                        required=False,
                        default=True)

    main(parser.parse_args())
