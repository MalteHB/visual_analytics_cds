
import argparse

from utils.utils import fetch_mnist

import numpy as np

# Import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def main(args):

    print("Initiating some awesome logistic regression classification!")

    # Importing arguments from the arguments parser

    random_state = args.rs

    test_size = args.ts

    scaling = args.s

    minmax = args.mm

    lr_mnist = LogisticRegressionMNIST()

    X_train, X_test, y_train, y_test = lr_mnist.split_and_preprocess_data(random_state=random_state,
                                                                          test_size=test_size,
                                                                          scaling=scaling,
                                                                          minmax=minmax)

    clf_model = lr_mnist.train(X_train, X_test, y_train, y_test)

    lr_mnist.print_eval_metrics(clf_model, X_test, y_test)

    print("DONE! Have a nice day. :-)")


class LogisticRegressionMNIST:

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

        return X_train, X_test, y_train, y_test

    def train(self, X_train, X_test, y_train, y_test):

        clf_model = LogisticRegression(penalty='none',
                                       tol=0.1,
                                       solver='saga',
                                       multi_class='multinomial').fit(X_train, y_train)

        return clf_model

    def print_eval_metrics(self, clf_model, X_test, y_test):
        """Prints the evaluation metrics to the terminal.

        Args:
            clf_model (sklearnModel): [description]
            X_test ([type]): [description]
            y_test ([type]): [description]
        """

        y_pred = clf_model.predict(X_test)

        cm = metrics.classification_report(y_test, y_pred)

        print(cm)


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

    main(parser.parse_args())
