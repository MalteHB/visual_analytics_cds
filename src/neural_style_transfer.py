import os
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from pathlib import Path
import numpy as np
import PIL.Image
import time
from utils.utils import setting_default_out_dir


def main(args):

    print("Initiating some awesome neural style transfer!")

    # Importing arguments from the arguments parser

    content_path = args.cp

    style_path = args.sp

    epochs = args.e

    out_dir = args.od

    gpu_device = args.gpu

    hub_model_link = args.hml

    pretrained = args.pretrained

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device  # Setting visible GPU devices

    nst = NeuralStyleTransfer(content_path=content_path,
                              style_path=style_path,
                              out_dir=out_dir,
                              hub_model_link=hub_model_link,
                              pretrained=pretrained)

    if not pretrained:  # Only create and train model if not using the pretrained model.

        nst.create_model()

        nst.train_model(epochs=epochs)

        nst.plot_loss()

    nst.save_image(out_dir=out_dir, pretrained=pretrained)

    print("DONE! Have a nice day. :-)")


class NeuralStyleTransfer:

    def __init__(self,
                 content_path=None,
                 style_path=None,
                 out_dir=None,
                 hub_model_link=None,
                 pretrained=False
                 ):

        self.content_path = content_path

        self.style_path = style_path

        self.out_dir = out_dir

        self.hub_model_link = hub_model_link

        if self.content_path is None:

            self.content_path = Path.cwd() / "data" / "NST" / "content.png"

        if self.style_path is None:

            self.style_path = Path.cwd() / "data" / "NST" / "style.png"

        if self.out_dir is None:

            self.out_dir = setting_default_out_dir()

        self.content_image, self.style_image = self.load_images(content_path=self.content_path,
                                                                style_path=self.style_path)  # Loading images

        if pretrained:

            if self.hub_model_link is None:

                self.hub_model_link = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'

            hub_model = hub.load(self.hub_model_link)

            self.image = hub_model(tf.constant(self.content_image), tf.constant(self.style_image))[0]

        return

    def load_images(self, content_path, style_path):
        """Loads and preprocesses images.

        Args:
            content_path (str): Path to the content image.
            style_path (str): Path to the style image.

        Returns:
            content_image (tf.Tensor): The content image.
            style_image (tf.Tensor): The style image.
        """

        # Reads and outputs the entire contents of the input filename.
        content_image = tf.io.read_file(str(content_path))

        style_image = tf.io.read_file(str(style_path))

        content_image = self._preprocess_image(content_image)

        style_image = self._preprocess_image(style_image)

        return content_image, style_image

    def create_model(self, content_image=None, style_image=None, content_layers=None, style_layers=None, content_weight=1e4, style_weight=1e-2, total_variation_weight=300):
        """Creates the Neural Style Transfer Model and initializes optimizers and different hyperparameters.

        Args:
            content_image (tf.Tensor, optional): The content image. Defaults to None.
            style_image (tf.Tensor, optional): The style image. Defaults to None.
            content_layers (list, optional): List with the names of the blocks and layers used for the content image. Defaults to None.
            style_layers (list, optional): List with the names of the blocks and layers used for the style image. Defaults to None.
            content_weight (float, optional): Weight for the content loss. Defaults to 1e4.
            style_weight (float, optional): Weight for the style loss. Defaults to 1e-2.
            total_variation_weight (int, optional): Weight for the total variation loss. Defaults to 300.
        """
        # If None use class image
        if content_image is None:

            content_image = self.content_image

        # If None use class image
        if style_image is None:

            style_image = self.style_image

        # If None use block 5 from the second convolution layer for content
        if content_layers is None:

            content_layers = ['block5_conv2']

        self.content_layers = content_layers

        # If None use block 1-5 from the first convolution layer for style
        if style_layers is None:

            style_layers = ['block1_conv1',
                            'block2_conv1',
                            'block3_conv1',
                            'block4_conv1',
                            'block5_conv1']

        self.style_layers = style_layers

        self.content_weight = content_weight

        self.style_weight = style_weight

        
        self.total_variation_weight = total_variation_weight  # To decrease high frequency artifacts total variaton loss is calculated by using a total variation loss

        self.model = StyleContentModel(style_layers, content_layers)

        # Setting style and content target values:
        self.style_targets = self.model(style_image)['style']

        self.content_targets = self.model(content_image)['content']

        # Using Adam as the optimizer. 
        self.opt = tf.optimizers.Adam(learning_rate=0.005, beta_1=0.99, epsilon=1e-1)

    def train_model(self, epochs=25, steps_per_epoch=100):
        """Trains the model using the content and style images belonging to the NeuralStyleTransfer class.

        Args:
            epochs (int, optional): Number of epochs. Defaults to 25.
            steps_per_epoch (int, optional): Training steps per epoch. Defaults to 100.
        """

        self.image = tf.Variable(self.content_image)  # Creating tensorflow variable.

        start = time.time()

        self.train_loss_results = []

        self.epochs = epochs

        self.steps_per_epoch = steps_per_epoch

        step = 0

        # Training loop
        for epoch in tqdm(range(self.epochs)):

            print(f"Epoch {self.epochs+1}")

            epoch_loss_avg = tf.keras.metrics.Mean()

            for epoch_step in tqdm(range(self.steps_per_epoch)):

                step += 1

                loss, _ = self._train_step(self.image)


                epoch_loss_avg.update_state(loss)  # Update avg epoch loss

            self.train_loss_results.append(epoch_loss_avg.result())  # Store loss for plot

        end = time.time()

        print(f"Total time: {end-start}")

    def plot_loss(self, out_dir=None):
        """Plots a curve of the training loss.

        Args:
            out_dir (str, optional): Path to save the loss plot. Defaults to None.
        """

        if out_dir is None:

            out_dir = self.out_dir

        out_path = out_dir / f"loss_{self.epochs}_epochs.png"

        # Creating matplotlob plot
        fig, axes = plt.subplots(1, sharex=True, figsize=(12, 8))
        fig.suptitle('Training Loss Full Training')

        axes.set_ylabel("Loss", fontsize=14)
        axes.plot(self.train_loss_results)

        axes.set_xlabel("Epoch", fontsize=14)

        plt.savefig(str(out_path))
        plt.show()

    def save_image(self, image=None, out_dir=None, pretrained=False):
        """Saves an image.

        Args:
            image (tf.Tensor, optional): Image to be saved. Defaults to None.
            out_dir (PosixPath, optional): Path to save the image to. Defaults to None.
            pretrained (bool, optional): Whether a pretrained Neural Style Transfer model was used. Defaults to False.
        """

        if image is None:

            image = self.image

        if out_dir is None:

            out_dir = self.out_dir

        if not pretrained:

            out_path = out_dir / f"stylized_image_{self.epochs}_epochs.png"

        else:

            out_path = out_dir / "pretrained_stylized_image.png"

        print(f"Saving image to: {out_path}")

        tf.keras.preprocessing.image.save_img(out_path, image[0])  # Saving image

    def _preprocess_image(self, image):
        """Preprocesses an image.

        Args:
            image (tf.Tensor): Image loaded through tf.io.

        Returns:
            image (tf.Tensor): A preprocessed image ready for training.
        """

        image = tf.image.decode_image(image, channels=3)  # Decoding the image

        image = tf.image.convert_image_dtype(image, tf.float32)  # Convert image to tf.float32.

        image = self._image_scaler(image)  # Scales the image.

        image = image[tf.newaxis, :]  # The model requires a four dimensional tensor, hence the addition of this.

        return image

    def _image_scaler(self, image, max_dim=256):
        """Scales an input image.

        Args:
            image (tf.Tensor): Image to scale.
            max_dim (int, optional): Max dimension for the values. Defaults to 256.

        Returns:
            image (tf.Tensor): A scaled image.
        """

        original_shape = tf.cast(tf.shape(image)[:-1], tf.float32)  # Casts a tensor to a new type.

        scale_ratio = 4 * max_dim / max(original_shape)  # Creates a scale constant for the image.

        new_shape = tf.cast(original_shape * scale_ratio, tf.int32)  # Casts a tensor to a new type.

        image = tf.image.resize(image, new_shape)  # Resizes the image based on the scaling constant generated above.

        return image

    def _style_content_loss(self, outputs):
        """Calculate style loss for the model

        Args:
            outputs (dict): Output dictionary containing the content and style outputs

        Returns:
            loss (): [description]
        """
        
        # Taking content and style outputs
        content_outputs = outputs['content']
        style_outputs = outputs['style']

        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name])**2)
                               for name in style_outputs.keys()])  # Calculate absolute style loss

        style_loss *= self.style_weight / len(self.style_layers)  # Assign weight to absolute loss

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name])**2)
                                for name in content_outputs.keys()])  # Calculate absolute content loss

        content_loss *= self.content_weight / len(self.content_layers)  # Assign weight to absolute loss

        loss = style_loss + content_loss  # Combine losses

        return loss

    @tf.function()
    def _train_step(self, image):
        """Training step for the model.

        Args:
            image (tf.Tensor): Content image to train on.

        Returns:
            loss(tf.Tensor): Loss of the model from the 'prediction'
            grad(tf.Tensor): Gradients from the loss and the image
        """
        with tf.GradientTape() as tape:
            outputs = self.model(image)
            loss = self._style_content_loss(outputs)
            loss += self.total_variation_weight * tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(tf.clip_by_value(image,
                                      clip_value_min=0.0,
                                      clip_value_max=1.0))

        return loss, grad


class StyleContentModel(tf.keras.models.Model):
    """Class for the style content model

    Args:
        tf (tf.keras.models.Model): Keras model
    """

    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()


        self.vgg = self._vgg_layers(style_layers + content_layers)  # Create VGG19 model with the chosen layers
        self.vgg.trainable = False  # It is not trainable

        # Used as keys in dict creation
        self.style_layers = style_layers
        self.content_layers = content_layers

    def call(self, inputs):
        """Creates the output dictionaries for the content and style image

        Args:
            inputs (list): Lists of style and content layers

        Returns:
            dict: Dictionary with content and style outputs
        """

        # Preprocessing the inputs
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.vgg(preprocessed_input) # Feed the preprocessed image to the VGG19 model

        style_outputs, content_outputs = (outputs[:len(self.style_layers)],
                                          outputs[len(self.style_layers):])  # Separate style and content outputs

        style_outputs = [self._gram_matrix(style_output) for style_output in style_outputs]  # Process style output before dict creation

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}  # Create two dicts for content and style outputs

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layers, style_outputs)}

        output_dict = {'content': content_dict, 'style': style_dict}  # Create a combined dict

        return output_dict

    def _vgg_layers(self, layer_names):
        """Creates a pre-trained VGG model which takes an input and returns a list of intermediate output values

        Args:
            layer_names (list): Names of the VGG19 layers to be used

        Returns:
            model(tf.keras.model): Keras model with the VGG19 layers
        """
        # Load the pretrained VGG19 and create it with only the assigned layers
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False
        outputs = [vgg.get_layer(name).output for name in layer_names]
        model = tf.keras.Model([vgg.input], outputs)
        return model

    def _gram_matrix(self, input_tensor):
        """Calculates tensor contraction over the outer product and the specified indices

        Args:
            input_tensor (tf.Tensor): Input tensor from the layers

        Returns:
            tf.Tensor: Gram matrix
        """

        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)  # Matrix multiplication

        input_shape = tf.shape(input_tensor)  # Save the shape of the input tensor

        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)  # Casts a tensor to a new type.

        gram_matrix = result / (num_locations) # Divide matrix multiplication output to num_locations

        return gram_matrix


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cp',
                        metavar="Content Path",
                        type=str,
                        help='Path to the content image.',
                        required=False)

    parser.add_argument('--sp',
                        metavar="Style Path",
                        type=str,
                        help='Path to the style image.',
                        required=False)

    parser.add_argument('--e',
                        metavar="Epochs",
                        type=int,
                        help='Number of epochs for the neural network training.',
                        required=False,
                        default=25)

    parser.add_argument('--od',
                        metavar="Output Directory",
                        type=str,
                        help='Path to the output directory.',
                        required=False)

    parser.add_argument('--gpu',
                        metavar="GPU device",
                        type=str,
                        help='Which GPU device to use. Defaults to -1 to not allow GPU usage.',
                        required=False,
                        default="-1")

    parser.add_argument('--hml',
                        metavar="Hub Model Link",
                        type=str,
                        help='Link to at Neural Style Transfer Model on the TensorFlow Hub.',
                        required=False)

    parser.add_argument('--pretrained',
                        dest="pretrained",
                        help='Whether to use a pretrained model or not to',
                        action="store_true")

    parser.set_defaults(pretrained=False)

    main(parser.parse_args())
