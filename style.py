import argparse
import scipy.misc
import scipy.io
import numpy as np
import tensorflow as tf

# Constants
VGG_MODEL = "imagenet-vgg-verydeep-19.mat"
ALPHA = 1
BETA = 1000
DIMENSIONS = 3

# Code to read command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--content",
                    help="Enter the content image",
                    dest="content_image",
                    type=str,
                    default="./Inputs/tubingen.jpg")

parser.add_argument("--style",
                    help="Enter the style image",
                    dest="style_image",
                    type=str,
                    default="./Inputs/starry_night.jpg")

parser.add_argument("--output",
                    help="Enter the output directory",
                    dest="output_directory",
                    type=str,
                    default="./Outputs")

parser.add_argument("--width",
                    help="Enter the image width",
                    dest="image_width",
                    type=int,
                    default=200)

parser.add_argument("--height",
                    help="Enter the image height",
                    dest="image_height",
                    type=int,
                    default=150)

arguments = parser.parse_args()
image_height = arguments.image_height
image_width = arguments.image_width


def initialize_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (image_height, image_width))
    return image


def convolution(prev_layer, weights):
    return tf.nn.conv2d(prev_layer, weights, strides=[1, 1, 1, 1], padding='SAME')


def get_weights(layer, i):
    weights = layer[i][0][0][0][0][0]
    weights = tf.constant(weights)
    return weights


def load_vgg(path):
    model = {}
    vgg_layers = scipy.io.loadmat(path)["layers"][0]

    model['input'] = tf.Variable(np.zeros((1, image_height, image_width, DIMENSIONS), dtype=np.float32))
    model['conv1_1'] = convolution(model['input'], get_weights(vgg_layers, 0))
    return model


def main():
    # Resize content and style images
    content_image = initialize_image(arguments.content_image)
    style_image = initialize_image(arguments.style_image)
    vgg_model = load_vgg(VGG_MODEL)

    # Generate a white noise image
    noise_image = np.random.random((image_height, image_width))


if __name__ == '__main__':
    main()
