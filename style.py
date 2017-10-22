import argparse
import scipy.misc
import scipy.io
import numpy as np
import tensorflow as tf

# Constants
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

parser.add_argument("--alpha",
                    help="Enter the weight factor for content",
                    dest="alpha",
                    type=int,
                    default=1)

parser.add_argument("--beta",
                    help="Enter the weight factor for style",
                    dest="beta",
                    type=int,
                    default=100)

parser.add_argument("--vgg",
                    help="Enter the VGG19 model",
                    dest="vgg",
                    type=str,
                    default="imagenet-vgg-verydeep-19.mat")

arguments = parser.parse_args()
image_height = arguments.image_height
image_width = arguments.image_width


def initialize_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (image_height, image_width))
    return image


def convolution(prev_layer, weights):
    return tf.nn.conv2d(prev_layer, weights, strides=(1, 1, 1, 1), padding="SAME")


def relu(prev_layer, bias):
    return tf.nn.relu(prev_layer + bias)


def avg_pool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")


def get_weights(layer, i):
    weights = layer[i][0][0][0][0][0]
    weights = tf.constant(weights)
    return weights


def get_bias(layer, i):
    bias = layer[i][0][0][0][0][1]
    bias = tf.constant(bias)
    return bias


def load_vgg(path):
    model = {}
    vgg_layers = scipy.io.loadmat(path)
    vgg_layers = vgg_layers["layers"][0]

    model["input"] = tf.Variable(np.zeros((1, image_height, image_width, DIMENSIONS),
                                          dtype=np.float32))

    # Group 1
    model["conv1_1"] = convolution(model["input"], get_weights(vgg_layers, 0))
    model["relu1_1"] = relu(model["conv1_1"], get_bias(vgg_layers, 2))
    model["conv1_2"] = convolution(model["relu1_1"], get_weights(vgg_layers, 2))
    model["relu1_2"] = relu(model["conv1_2"], get_bias(vgg_layers, 2))
    model["avg_pool1"] = avg_pool(model["relu1_2"])

    # Group 2
    model["conv2_1"] = convolution(model["avg_pool1"], get_weights(vgg_layers, 5))
    model["relu2_1"] = relu(model["conv2_1"], get_bias(vgg_layers, 5))
    model["conv2_2"] = convolution(model["relu2_1"], get_weights(vgg_layers, 7))
    model["relu2_2"] = relu(model["conv2_2"], get_bias(vgg_layers, 7))
    model["avg_pool2"] = avg_pool(model["relu2_2"])

    # Group 3
    model["conv3_1"] = convolution(model["avg_pool2"], get_weights(vgg_layers, 10))
    model["relu3_1"] = relu(model["conv3_1"], get_bias(vgg_layers, 10))
    model["conv3_2"] = convolution(model["relu3_1"], get_weights(vgg_layers, 12))
    model["relu3_2"] = relu(model["conv3_2"], get_bias(vgg_layers, 12))
    model["conv3_3"] = convolution(model["relu3_2"], get_weights(vgg_layers, 14))
    model["relu3_3"] = relu(model["conv3_3"], get_bias(vgg_layers, 14))
    model["conv3_4"] = convolution(model["relu3_3"], get_weights(vgg_layers, 16))
    model["relu3_4"] = relu(model["conv3_4"], get_bias(vgg_layers, 16))
    model["avg_pool3"] = avg_pool(model["relu3_4"])

    # Group 4
    model["conv4_1"] = convolution(model["avg_pool3"], get_weights(vgg_layers, 19))
    model["relu4_1"] = relu(model["conv4_1"], get_bias(vgg_layers, 19))
    model["conv4_2"] = convolution(model["relu4_1"], get_weights(vgg_layers, 21))
    model["relu4_2"] = relu(model["conv4_2"], get_bias(vgg_layers, 21))
    model["conv4_3"] = convolution(model["relu4_2"], get_weights(vgg_layers, 23))
    model["relu4_3"] = relu(model["conv4_3"], get_bias(vgg_layers, 23))
    model["conv4_4"] = convolution(model["relu4_3"], get_weights(vgg_layers, 25))
    model["relu4_4"] = relu(model["conv4_4"], get_bias(vgg_layers, 25))
    model["avg_pool4"] = avg_pool(model["relu4_4"])

    # Group 5
    model["conv5_1"] = convolution(model["avg_pool4"], get_weights(vgg_layers, 28))
    model["relu5_1"] = relu(model["conv5_1"], get_bias(vgg_layers, 28))
    model["conv5_2"] = convolution(model["relu5_1"], get_weights(vgg_layers, 30))
    model["relu5_2"] = relu(model["conv5_2"], get_bias(vgg_layers, 30))
    model["conv5_3"] = convolution(model["relu5_2"], get_weights(vgg_layers, 32))
    model["relu5_3"] = relu(model["conv5_3"], get_bias(vgg_layers, 32))
    model["conv5_4"] = convolution(model["relu5_3"], get_weights(vgg_layers, 34))
    model["relu5_4"] = relu(model["conv5_4"], get_bias(vgg_layers, 34))
    model["avg_pool5"] = avg_pool(model["relu5_4"])

    return model


def main():
    # Resize content and style images
    content_image = initialize_image(arguments.content_image)
    style_image = initialize_image(arguments.style_image)

    # Load layers of the VGG model
    vgg_model = load_vgg(arguments.vgg)

    # Generate a white noise image
    noise_image = np.random.random((image_height, image_width))


if __name__ == "__main__":
    main()
