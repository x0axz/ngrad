# Autograd Engine

The repository includes an Autograd engine and a neural network library that handles N-dimensional arrays.

Autograd is a tool used for derivative calculation. It tracks operations on values with enabled gradients and builds a dynamic computational graph — a graph without cycles. Input values serve as the leaves of the graph, while output values act as its roots. Gradients are computed by traversing the graph from root to leaf, applying the chain rule to multiply gradients at each step.

Andrej Karaphy's [Micrograd](https://github.com/karpathy/micrograd) served as inspiration for this project. But this Autograd engine will accept N-dimensional arrays, whereas Microgard accepts scalar values only.

## Blog

[Building an Autograd Engine: An Illustrative and Interactive Guide](https://x0axz.com/blog/autograd.html)

The article provides a comprehensive guide to building an autograd engine. It assumes a basic understanding of Python programming, high school calculus, and neural networks but offers various teaching methods for beginners. It includes line-by-line code explanations, output visualizations, and an interactive area to explore derivatives. The guide covers the foundational concepts of neural networks, starting with derivatives and progressing to backpropagation. It explains how to perform backpropagation manually and programmatically, including implementation techniques. The article also demonstrates the building of an autograd class from scratch and its application to training a neural network on a dataset. It concludes by guiding readers through the development of a simple neural network library using the autograd class.
