# Autograd Engine

The repository includes an Autograd engine and a neural network library that handle an N-dimensional array.

Autograd is a tool used for derivative calculation. It tracks operations on values with enabled gradients and builds a dynamic computational graph — a graph without cycles. Input values serve as the leaves of the graph, while output values act as its roots. Gradients are computed by traversing the graph from root to leaf, applying the chain rule to multiply gradients at each step.

Andrej Karaphy's [Micrograd](https://github.com/karpathy/micrograd) served as inspiration for this project. But this Autograd engine will accept N-dimensional array, whereas Microgard accepts scalar values only.

## Blog

[Building an Autograd Engine: An Illustrative and Interactive Guide](https://x0axz.com/blog/autograd.html)

The article provides a comprehensive guide to building an autograd engine and a neural network library that handle an N-dimensional array. It assumes a basic understanding of Python programming, high school calculus, and neural networks but offers various teaching methods for beginners. It includes line-by-line code explanations, output visualizations, and an interactive area to explore derivatives. The guide covers the foundational concepts of neural networks, starting with derivatives and progressing to backpropagation. It explains how to perform backpropagation manually and programmatically, including implementation techniques. The article also demonstrates the building of an autograd class from scratch and its application to training a neural network on a dataset. It concludes by guiding readers through the development of a simple neural network library using the autograd class.

## Installation

The only library required for this to work is Numpy, which is used to handle N-dimensional array.

```
pip install -r requirements.txt
```

## Test

To verify the accuracy of the [engine.py](https://github.com/x0axz/ngrad/blob/main/ngrad/engine.py) code and ensure that it produces the same output for the N-dimensional array, scalar value, and PyTorch APIs, execute the following command:

```
cd test/
python engine_test.py
```

## Train

This [notebook](https://github.com/x0axz/ngrad/blob/main/notebook/Training_Neural_Network.ipynb) comprises the code required for training the neural network, encompassing the code for [engine.py](https://github.com/x0axz/ngrad/blob/main/ngrad/engine.py) and [library.py](https://github.com/x0axz/ngrad/blob/main/ngrad/library.py). Within this notebook, we first set the input values, construct an MLP with a predetermined architecture, and subsequently execute forward propagation to compute and acquire the output. Subsequently, we initiate the training procedure of the neural network by generating a small dataset, followed by training the MLP model to minimize loss and enhance its prediction abilities. Finally, we present a list of the predicted outputs.

## Notebook

This [notebook](https://github.com/x0axz/ngrad/blob/main/notebook/Autograd_Engine_&_NN_Library.ipynb) contains a comprehensive collection of examples and code for building, testing, and training the autograd engine and neural network library. Play with it to experiment and explore its functionalities.
