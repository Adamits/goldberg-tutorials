"""
This uses classes defined in the file nn.py

We are mainly using the Layer class, which defines matrix operations
for the forward and backward pass at the instance of a single layer,
and its relationship to a 'child' layer. It would be useful to look at
how these matrices are instantiated, and how their operations are implemented.
"""
from nn import *
import numpy as np

# Xavier initialization, the 4 is from d_in + d_out
init = np.sqrt(6) / np.sqrt(4)

# Input layer, each sample will have 2 dimensions
il = InputLayer(size=2)
# Hidden Layer
hl = Layer(size=3, child_layer=il, init=init)
# Output Layer; size should be the number of potential class layers
ol = Layer(size=2, child_layer=hl, init=init)
# Loss Layer
ll = LossLayer(child_layer=ol)

# These are the XOR values. each tuple has (coordinate pairs, one hot vector for label)
input_samples = [([1, 1], [1, 0]), ([-1, 1], [0, 1]), ([-1, -1], [1, 0]), ([1, -1], [0, 1])]

# Run 20 epochs
for e in range(20):
  """
  SGD, but in practice we will actually just sample everything
  """
  # For tracking losses in a single epoch
  losses = []
  for inp, target in input_samples:
    # Forward propogate each layer. Note the different forward implementation per layer type
    il.forward(inp)
    hl.forward()
    ol.forward()
    ll.forward(target)
    # Track per isntance loss
    losses.append(ll.loss)

    # Now backward propogate the error starting at the loss node. Note the different forward implementation per layer type
    ll.backward(target)
    ol.backward()
    hl.backward()

  print("AVG LOSS: %.2f" % (sum(losses) / len(losses)))


