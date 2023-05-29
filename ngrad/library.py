from ngrad.engine import Value
import random

class Neuron:

  def __init__(self, nin):
    # Initialize the neuron with random weights and bias between -1 and 1
    self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
    self.b = Value(random.uniform(-1,1))
  
  def __call__(self, x):
    # Calculate the weighted sum of inputs multiplied by weights and add the bias
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    # Apply the hyperbolic tangent function to the activation
    out = act.tanh()
    # Return the output of the neuron
    return out
  
  def parameters(self):
    # Return the weights and bias of the neuron as parameters
    return self.w + [self.b]
  
class Layer:

  def __init__(self, nin, nout):
    # Create a layer with a specified number of input and output neurons
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    # Compute the output of each neuron in the layer given an input
    outs = [n(x) for n in self.neurons]
    # If there is only one output neuron, return it directly; otherwise, return a list of outputs
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    # Return the parameters of all neurons in the layer
    return [p for neuron in self.neurons for p in neuron.parameters()]
  
class MLP:

  def __init__(self, nin, nouts):
    # Create a multi-layer perceptron (MLP) with the specified number of input and output neurons for each layer
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

  def __call__(self, x):
    # Forward propagate the input through all layers of the MLP
    for layer in self.layers:
      x = layer(x)
    # Return the final output of the MLP
    return x

  def parameters(self):
    # Return the parameters of all layers in the MLP
    return [p for layer in self.layers for p in layer.parameters()]
