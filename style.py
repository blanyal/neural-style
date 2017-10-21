import argparse

# Constants
VGG_MODEL = "imagenet-vgg-verydeep-19.mat"

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

