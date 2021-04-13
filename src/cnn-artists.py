import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from sklearn.metrics import classification_report

from utils.utils import setting_default_data_dir, setting_default_out_dir


def main(args):

    print("Initiating some awesome convolutional neural networks!")

    # Importing arguments from the arguments parser

    random_state = args.rs

    train_data_dir = args.tdd

    val_data_dir = args.vdd

    batch_size = args.bs

    img_height = args.img_h

    img_width = args.img_w

    cnn = CNNClassification()

    cnn.preprocess_data(train_data_dir=train_data_dir,
                        val_data_dir=val_data_dir,
                        img_height=img_height,
                        img_width=img_width,
                        batch_size=batch_size,
                        random_state=random_state)

    cnn.create_model(img_height=img_height,
                     img_width=img_width)

    cnn.train()

    cnn.plot_training()

    cnn.evaluate_model()

    print("DONE! Have a nice day. :-)")


class CNNClassification:

    def __init__(self):
        return

    def preprocess_data(self, train_data_dir, val_data_dir, img_height, img_width, batch_size, random_state=1):
        """Preprocesses the data from the directories into TensorFlow Dataset objects that can be used directly with the Tensorflow Keras Sequential API.

        Args:
            train_data_dir (PosixPath): Path to the training directory.
            val_data_dir (PosixPath): Path to the validation directory.
            img_height (int): Height of each image to use for rescaling.
            img_width (int): Width of each image to use for rescaling.
            batch_size (int): Batch size to use when loading data.
            random_state (int, optional): Random state to use when shuffling. Defaults to 1.
        """
        self.train_data_dir = train_data_dir

        self.val_data_dir = val_data_dir

        if train_data_dir is None:

            self.train_data_dir, _ = setting_default_data_dir(assignment=5)

        if val_data_dir is None:

            _, self.val_data_dir = setting_default_data_dir(assignment=5)

        self.out_dir = setting_default_out_dir()

        self.out_file_path = self.out_dir / "classification_report.txt"

        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(self.train_data_dir,
                                                                            seed=random_state,
                                                                            image_size=(img_height, img_width),
                                                                            batch_size=batch_size)

        self.val_ds = tf.keras.preprocessing.image_dataset_from_directory(self.val_data_dir,
                                                                          seed=random_state,
                                                                          image_size=(img_height, img_width),
                                                                          batch_size=batch_size)

        # Preprocessing data for evaluation and classification report

        self.val_images = np.concatenate([images for images, labels in self.val_ds], axis=0)

        self.val_labels = np.concatenate([labels for images, labels in self.val_ds], axis=0)

        self.train_class_names = self.train_ds.class_names

        self.val_class_names = self.val_ds.class_names

        self.num_classes = len(self.train_class_names)

        # Setting up dataset performance configurations

        AUTOTUNE = tf.data.AUTOTUNE  # Letting tensorflow autotune the buffer size

        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)  # Caching and shuffling of training data

        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)  # Caching validaton data

    def create_model(self, img_height, img_width, print_model=True):

        self.model = Sequential([layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
                                 layers.Conv2D(16, 3, padding='same', activation='relu'),
                                 layers.MaxPooling2D(),
                                 layers.Conv2D(32, 3, padding='same', activation='relu'),
                                 layers.MaxPooling2D(),
                                 layers.Conv2D(64, 3, padding='same', activation='relu'),
                                 layers.MaxPooling2D(),
                                 layers.Flatten(),
                                 layers.Dense(128, activation='relu'),
                                 layers.Dense(self.num_classes)
                                 ])

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        if print_model:

            self.model.summary()

    def train(self, epochs=5):

        self.epochs = epochs

        self.history = self.model.fit(self.train_ds,
                                      validation_data=self.val_ds,
                                      epochs=self.epochs
                                      )

    def plot_training(self, history=None):

        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs_range = range(self.epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        out_file = self.out_dir / "train_val_history.png"
        plt.savefig(out_file)

    def evaluate_model(self, batch_size=32):

        predictions = self.model.predict(self.val_images, batch_size=batch_size)

        eval_report = classification_report(self.val_labels,
                                            predictions.argmax(axis=1),
                                            target_names=self.val_class_names)

        print(eval_report)

        out_file = self.out_dir / "classification_report.txt"

        out_file.write_text(eval_report)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--rs',
                        metavar="Random State",
                        type=int,
                        help='Random State of the logistic regression model.',
                        required=False,
                        default=1)

    parser.add_argument('--tdd',
                        metavar="Train Data Directory",
                        type=str,
                        help='Path to the training data',
                        required=False)

    parser.add_argument('--vdd',
                        metavar="Validation Data Directory",
                        type=str,
                        help='Path to the validation data',
                        required=False)

    parser.add_argument('--bs',
                        metavar="Batch Size",
                        type=int,
                        help='The batch size of the model.',
                        required=False,
                        default=32)

    parser.add_argument('--e',
                        metavar="Epochs",
                        type=int,
                        help='Number of epochs for the neural network training.',
                        required=False,
                        default=10)

    parser.add_argument('--img_h',
                        metavar="Image Height",
                        type=int,
                        help='Pixel height for image rescaling.',
                        required=False,
                        default=256)

    parser.add_argument('--img_w',
                        metavar="Image Width",
                        type=int,
                        help='Pixel width for image rescaling.',
                        required=False,
                        default=256)

    main(parser.parse_args())
