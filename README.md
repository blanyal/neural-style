# neural-style
TensorFlow implementation of [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

This project is about an artificial system based on a Deep Neural Network that creates artistic images using TensorFlow's Machine Learning APIs.

## Examples
 - Style Image
<img src="https://github.com/blanyal/neural-style/blob/master/Inputs/starry_night.jpg" width="400">

 - Content Image
<img src="https://github.com/blanyal/neural-style/blob/master/Inputs/tubingen.jpg" width="400">

 - Output Image
<img src="https://github.com/blanyal/neural-style/blob/master/Outputs/output_image.jpg" width="400">

## Requirements
 - TensorFlow
 - SciPy
 - NumPy
 - [VGG 19 model](http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat)
 
## Usage
**Basic usage**:
```
python style.py
```
**Options**:
* `--content`: Name of the content image
* `--style`: Name of the style image
* `--output`: Name of the output directory
* `--width`: Width of the output image
* `--height`: Height of the output image
* `--alpha`: Weight factor for content
* `--beta`: Weight factor for style
* `--vgg`: Name of the VGG 19 model
* `--iterations`: Number of iterations
 
 ## References
 - https://github.com/ckmarkoh/neuralart_tensorflow
 - https://github.com/log0/neural-style-painting

## License
    MIT License

    Copyright (c) 2017 Blanyal D'Souza

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
