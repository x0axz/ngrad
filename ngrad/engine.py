import numpy as np

class Value:
  
  def __init__(self, data, _children=(), _op='', label=''):
    # Convert data to a NumPy array
    if isinstance(data, np.ndarray):
      self.data = data
    else:
      self.data = np.array(data)
      
    # Initialize the gradient as an array of zeros with the same shape as the data
    self.grad = np.zeros_like(self.data)
    
    # Placeholder for the backward function
    self._backward = lambda: None
    
    # Set of previous nodes (parents)
    self._prev = set(_children)
    
    # Operation associated with the current node
    self._op = _op
    
    # Optional label for the node
    self.label = label

  
  def __add__(self, other):
    # Convert other to a Value if it is not already
    other = other if isinstance(other, Value) else Value(other)
    
    # Create a new Value instance representing the addition operation
    out = Value(self.data + other.data, (self, other), '+')

    def _backward():
      # Calculate the gradients using the chain rule and update the gradients of the operands
      self.grad = self.grad + np.multiply(1.0, out.grad)
      other.grad = other.grad + np.multiply(1.0, out.grad)

    # Assign the backward function to the new Value instance
    out._backward = _backward
    return out


  def __mul__(self, other):
    # Convert other to a Value if it is not already
    other = other if isinstance(other, Value) else Value(other)
    
    # Create a new Value instance representing the multiplication operation
    out = Value(self.data * other.data, (self, other), '*')

    def _backward():
      # Calculate the gradients using the chain rule and update the gradients of the operands
      self.grad = self.grad + np.multiply(other.data, out.grad)
      other.grad = other.grad + np.multiply(self.data, out.grad)

    # Assign the backward function to the new Value instance
    out._backward = _backward
    return out


  def __pow__(self, other):
    if isinstance(other, (int, float)):
      # If the exponent is a scalar, perform element-wise power operation
      out_data = np.power(self.data, other)
      out = Value(out_data, (self,), f'**{other}')

      def _backward():
        # Calculate the gradients using the chain rule and update the gradients of the operand
        self.grad = self.grad + np.multiply(np.multiply(other, np.power(self.data, other - 1)), out.grad)

      out._backward = _backward
      return out
    elif isinstance(other, Value):
      # If the exponent is a Value instance, perform element-wise power operation
      out_data = np.power(self.data, other.data)
      out = Value(out_data, (self, other), f'**')

      def _backward():
        # Calculate the gradients using the chain rule and update the gradients of the operands
        self.grad = self.grad + np.multiply(np.multiply(other.data, np.power(self.data, other.data - 1)), out.grad)
        other.grad = other.grad + np.multiply(np.log(self.data), out.grad)

      out._backward = _backward
      return out
    else:
      raise TypeError("Unsupported operand type(s) for **: 'Value' and '{}'".format(type(other).__name__))

  def __radd__(self, other):
    # Perform right addition by the Value instance
    return np.add(self, other)

  def __rmul__(self, other):
    # Perform right multiplication by the Value instance
    return np.multiply(self, other)


  def __truediv__(self, other):
    # Perform true division by the Value instance
    return np.multiply(self, other**-1)


  def __neg__(self):
    # Perform negation of the Value instance
    return np.multiply(self, -1)


  def __sub__(self, other):
    # Perform subtraction of a Value instance
    return np.add(self, (-other))


  def exp(self):
    # Compute the element-wise exponential of the Value instance
    x = self.data
    out = Value(np.exp(x), (self, ), 'exp')

    def _backward():
        # Calculate the gradients using the chain rule
        self.grad = self.grad + np.multiply(out.data, out.grad)

    # Assign the backward function to the new Value instance
    out._backward = _backward
    return out


  def tanh(self):
    # Compute the element-wise hyperbolic tangent of the Value instance
    x = self.data
    t = np.tanh(x)
    out = Value(t, (self, ), 'tanh')

    def _backward():
        # Calculate the gradients using the chain rule
        self.grad = self.grad + np.multiply((1 - t**2), out.grad)

    # Assign the backward function to the new Value instance
    out._backward = _backward
    return out


  def backward(self):
    # Perform backpropagation to compute gradients
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)

    # Build the topological order of nodes
    build_topo(self)

    # Set the gradient of the output to ones (assuming scalar loss)
    self.grad = np.ones_like(self.data)

    # Perform backward pass through the nodes in reverse topological order
    for node in reversed(topo):
        node._backward()


  def __repr__(self):
    # String representation of the Value instance
    return f"Value(data={self.data})"
