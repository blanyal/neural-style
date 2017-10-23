import argparse
import scipy.misc
import scipy.io
import numpy as np
import tensorflow as tf
import os

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
                    default=800)

parser.add_argument("--height",
                    help="Enter the image height",
                    dest="image_height",
                    type=int,
                    default=600)

parser.add_argument("--alpha",
                    help="Enter the weight factor for content",
                    dest="alpha",
                    type=int,
                    default=10)

parser.add_argument("--beta",
                    help="Enter the weight factor for style",
                    dest="beta",
                    type=int,
                    default=1000)

parser.add_argument("--vgg",
                    help="Enter the VGG19 model",
                    dest="vgg",
                    type=str,
                    default="imagenet-vgg-verydeep-19.mat")

parser.add_argument("--iterations",
                    help="Enter the number of iterations",
                    dest="iterations",
                    type=int,
                    default=1000)

arguments = parser.parse_args()
image_height = arguments.image_height
image_width = arguments.image_width


# Function to resize input image and store it as an array
def initialize_image(path):
    image = scipy.misc.imread(path)
    image = scipy.misc.imresize(image, (image_height, image_width))
    image = image[np.newaxis, :, :, :]
    return image


# Function to save the output image at each interval
def save_image(path, image):
    scipy.misc.imsave(path, image[0])


# Function for convolution using weights
def convolution(prev_layer, weights):
    return tf.nn.conv2d(prev_layer, weights, strides=(1, 1, 1, 1), padding="SAME")


# Function for relu using bias
def relu(prev_layer, bias):
    return tf.nn.relu(prev_layer + bias)


# Function to apply average pooling as suggested in the paper
def avg_pool(prev_layer):
    return tf.nn.avg_pool(prev_layer, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="SAME")


# Function to get weights to be used for convolution
def get_weights(layer, i):
    weights = layer[i][0][0][0][0][0]
    weights = tf.constant(weights)
    return weights


# Function to get bias to be used for relu
def get_bias(layer, i):
    bias = layer[i][0][0][0][0][1]
    bias = tf.constant(bias)
    return bias


# Function for loss of content image
def calc_content_loss(p, x):
    # Equation 1 of the paper
    return 0.5 * tf.reduce_sum(tf.pow(x - p, 2))


# Function for loss of style image
def calc_style_loss(a, x):
    M = a.shape[1] * a.shape[2]
    N = a.shape[3]

    # Gram matrix of original image
    A = gram_matrix(a, M, N)

    # Gram matrix of generated image
    G = gram_matrix(x, M, N)

    # Equation 4 of the paper
    loss = (1 / (4 * (N ^ 2) * (M ^ 2))) * tf.reduce_sum(tf.pow((G - A), 2))
    return loss


# Function to get the gram matrix used to calculate style loss
def gram_matrix(x, area, depth):
    # Equation 3 of the paper
    F = tf.reshape(x, (area, depth))
    G = tf.matmul(tf.transpose(F), F)
    return G


# Load the VGG 19 network
def load_vgg(path):
    model = {}
    vgg_mat = scipy.io.loadmat(path)
    vgg_layers = vgg_mat["layers"][0]

    model["input"] = tf.Variable(np.zeros((1, image_height, image_width, 3),
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


# Main function
def main():
    # Resize content and style images
    content_image = initialize_image(arguments.content_image)
    style_image = initialize_image(arguments.style_image)

    # Load layers of the VGG model
    vgg_model = load_vgg(arguments.vgg)

    # Initialize TensorFlow
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Calculate the content loss using 'conv4_2' as suggested in the paper
        sess.run(vgg_model["input"].assign(content_image))
        content_loss = calc_content_loss(sess.run(vgg_model["conv4_2"]), vgg_model["conv4_2"])

        # Calculate style loss using the layers mentioned in the paper
        style_loss = 0
        layers = [("conv1_1", 1), ("conv2_1", 2), ("conv3_1", 3), ("conv4_1", 4), ("conv5_1", 5)]
        sess.run(vgg_model["input"].assign(style_image))

        for layer in layers:
            E = calc_style_loss(sess.run(vgg_model[layer[0]]), vgg_model[layer[0]])
            W = layer[1]

            # Equation 5 of the paper
            style_loss = style_loss + E * W

        # Get the content and style weight factors
        alpha = arguments.alpha
        beta = arguments.beta

        # Equation 7 of the paper
        total_loss = (alpha * content_loss) + (beta * style_loss)

        # Check if output directory exits. If it doesn't then create it
        output_directory = arguments.output_directory
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        # Train the network using L-BFGS optimizer
        iterations = arguments.iterations
        train_step = tf.contrib.opt.ScipyOptimizerInterface(
            total_loss,
            method="L-BFGS-B",
            options={"maxiter": iterations,
                     "disp": 100})

        sess.run(tf.global_variables_initializer())
        sess.run(vgg_model["input"].assign(content_image))
        train_step.minimize(sess)

        # Save the final image
        output_image = sess.run(vgg_model["input"])
        filename = output_directory + "/output_image.jpg"
        save_image(filename, output_image)


if __name__ == "__main__":
    main()
