from ngrad.engine import Value

#################################################################################

# Compute a forward pass on an N-dimensional array and perform backward propagation

# inputs x1, x2
x1 = Value([[2.0, 2.0], [2.0, 2.0]], label="x1")
x2 = Value([[0.0, 0.0], [0.0, 0.0]], label="x2")

# weights w1, w2
w1 = Value([[-3.0, -3.0], [-3.0, -3.0]], label="w1")
w2 = Value([[1.0, 1.0], [1.0, 1.0]], label="w2")

# bias of the neurons
b = Value([[6.8813735870195432, 6.8813735870195432], [6.8813735870195432, 6.8813735870195432]], label="b")

# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1 + x2w2'
n = x1w1x2w2 + b; n.label = 'n'

# apply the hyperbolic tangent function to n
o = n.tanh(); o.label = 'o'

print ("# N-dimensional array #")

print("Forward pass: ", o)

# perform backward propagation to compute gradients
o.backward()

print("Backward pass: ", o)

#################################################################################

# Compute a forward pass on scalar values and perform backward propagation



#################################################################################

# Compute a forward pass and perform backward propagation on the tanh function using PyTorch



#################################################################################
