from ngrad.engine import Value
import torch

####################################################################################################

# Compute a forward pass on an N-dimensional array and perform backward propagation

# inputs nd_x1, nd_x2
nd_x1 = Value([[2.0, 2.0], [2.0, 2.0]], label="nd_x1")
nd_x2 = Value([[0.0, 0.0], [0.0, 0.0]], label="nd_x2")

# weights nd_w1, nd_w2
nd_w1 = Value([[-3.0, -3.0], [-3.0, -3.0]], label="nd_w1")
nd_w2 = Value([[1.0, 1.0], [1.0, 1.0]], label="nd_w2")

# bias of the neurons
nd_b = Value([[6.8813735870195432, 6.8813735870195432], [6.8813735870195432, 6.8813735870195432]], label="nd_b")

# nd_x1*nd_w1 + nd_x2*nd_w2 + nd_b
nd_x1nd_w1 = nd_x1*nd_w1; nd_x1nd_w1.label = 'nd_x1*nd_w1'
nd_x2nd_w2 = nd_x2*nd_w2; nd_x2nd_w2.label = 'nd_x2*nd_w2'
nd_x1nd_w1nd_x2nd_w2 = nd_x1nd_w1 + nd_x2nd_w2; nd_x1nd_w1nd_x2nd_w2.label = 'nd_x1nd_w1 + nd_x2nd_w2'
nd_n = nd_x1nd_w1nd_x2nd_w2 + nd_b; nd_n.label = 'nd_n'

# apply the hyperbolic tangent function to n
nd = nd_n.tanh(); nd.label = 'nd'

print("#########################")
print("N-dimensional array")
print("#########################")

nd_fp = round(nd.data[0][0].item(), 5)

print("Forward pass: ", nd_fp)

# perform backward propagation to compute gradients
nd.backward()

nd_bp_x1 = round(nd_x1.grad[0][0].item(), 5)
nd_bp_x2 = round(nd_x2.grad[0][0].item(), 5)
nd_bp_w1 = round(nd_w1.grad[0][0].item(), 5)
nd_bp_w2 = round(nd_w2.grad[0][0].item(), 5)

print("Backward pass:")
print("nd_x1: ", nd_bp_x1)
print("nd_x2: ", nd_bp_x2)
print("nd_w1: ", nd_bp_w1)
print("nd_w2: ", nd_bp_w2)

####################################################################################################

# Compute a forward pass on scalar values and perform backward propagation

# inputs sv_x1, sv_x2
sv_x1 = Value(2.0, label="sv_x1")
sv_x2 = Value(0.0, label="sv_x2")

# weights sv_w1, sv_w2
sv_w1 = Value(-3.0, label="sv_w1")
sv_w2 = Value(1.0, label="sv_w2")

# bias of the neurons
sv_b = Value(6.8813735870195432, label="sv_b")

# sv_x1*sv_w1 + sv_x2*sv_w2 + sv_b
sv_x1sv_w1 = sv_x1*sv_w1; sv_x1sv_w1.label = 'sv_x1*sv_w1'
sv_x2sv_w2 = sv_x2*sv_w2; sv_x2sv_w2.label = 'sv_x2*sv_w2'
sv_x1sv_w1sv_x2sv_w2 = sv_x1sv_w1 + sv_x2sv_w2; sv_x1sv_w1sv_x2sv_w2.label = 'sv_x1sv_w1 + sv_x2sv_w2'
sv_n = sv_x1sv_w1sv_x2sv_w2 + sv_b; sv_n.label = 'sv_n'

# break tanh() into the following expression (hyperbolic tangent)

# computes the value of 'sv_e' by taking the exponential function of the expression '2*sv_n'
sv_e = (2*sv_n).exp(); sv_e.label = 'sv_e'
# computes the value of 'sv' using the values of 'sv_e' & subtracts 1 from 'sv_e' and divides it by 'sv_e' plus 1, following a mathematical expression
sv = (sv_e - 1) / (sv_e + 1); sv.label = 'sv'

print("#########################")
print("Scalar value")
print("#########################")

sv_fp = round(sv.data.item(), 5)

print("Forward pass: ", sv_fp)

# perform backward propagation to compute gradients
sv.backward()

sv_bp_x1 = round(sv_x1.grad.item(), 5)
sv_bp_x2 = round(sv_x2.grad.item(), 5)
sv_bp_w1 = round(sv_w1.grad.item(), 5)
sv_bp_w2 = round(sv_w2.grad.item(), 5)

print("Backward pass:")
print("sv_x1: ", sv_bp_x1)
print("sv_x2: ", sv_bp_x2)
print("sv_w1: ", sv_bp_w1)
print("sv_w2: ", sv_bp_w2)

####################################################################################################

# Compute a forward pass and perform backward propagation on the tanh function using PyTorch

# inputs pyt_x1, pyt_x2
pyt_x1 = torch.Tensor([2.0]); pyt_x1.requires_grad = True 
pyt_x2 = torch.Tensor([-.0]); pyt_x2.requires_grad = True

# weights pyt_w1, pyt_w2
pyt_w1 = torch.Tensor([-3.0]); pyt_w1.requires_grad = True  
pyt_w2 = torch.Tensor([1.0]); pyt_w2.requires_grad = True  

# bias of the neurons
pyt_b = torch.Tensor([6.8813735870195432]); pyt_b.requires_grad = True

# perform the computation: pyt_n = pyt_x1*pyt_w1 + pyt_x2*pyt_w2 + pyt_b
pyt_n = pyt_x1*pyt_w1 + pyt_x2*pyt_w2 + pyt_b

# apply the hyperbolic tangent function to pyt_n
pyt = torch.tanh(pyt_n)

print("#########################")
print ("Pytorch")
print("#########################")

pyt_fp = round(pyt.data.item(), 5)

print("Forward pass: ", pyt_fp)

# perform backward propagation to compute gradients
pyt.backward()

pyt_bp_x1 = round(pyt_x1.grad.item(), 5)
pyt_bp_x2 = round(pyt_x2.grad.item(), 5)
pyt_bp_w1 = round(pyt_w1.grad.item(), 5)
pyt_bp_w2 = round(pyt_w2.grad.item(), 5)

print("Backward pass:")
print("pyt_x1: ", pyt_bp_x1)
print("pyt_x2: ", pyt_bp_x2)
print("pyt_w1: ", pyt_bp_w1)
print("pyt_w2: ", pyt_bp_w2)

####################################################################################################

print("#########################")

if (nd_fp == sv_fp == pyt_fp):
    print("Forward Pass - Test Passed!")
else:
    print("Forward Pass - Test Failed!")

if ((nd_bp_x1 == sv_bp_x1 == pyt_bp_x1) and (nd_bp_x2 == sv_bp_x2 == pyt_bp_x2) and (nd_bp_w1 == sv_bp_w1 == pyt_bp_w1) and (nd_bp_w2 == sv_bp_w2 == pyt_bp_w2)):
    print("Backward Pass - Test Passed!")
else:
    print("Backward Pass - Test Failed!")

print("#########################")
