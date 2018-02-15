import numpy as np

class NN():
  def __init__(size_in, num_hidden_layers, size_hidden, size_out):
    self.input_size = input_size
    self.num_hidden_layers = hidden_layers
    self.hidden_size = hidden_size
    self.output_size = output_size

    self.input_layer = InputLayer(self.input_size)
    self.hidden_layers = []
    # Instantiate the hidden layers
    for i in range(hidden_layers):
      if i == 0:
        # input is the child for the first hidden layer
        self.hidden_layers.append(Layer(self.hidden_size, self.input_layer))
      else:
        # Previous hidden layer is the child for each subsequent hidden layer
        self.hidden_layers.append(Layer(self.hidden_size, self.hidden_layers[i-1]))
    # Set the output layer with last hidden layer as its child
    self.output_layer = Layer(output_size, hidden_layers[-1])

    # instantiate blank LR, can be set in the train() method
    self.lr = 0.0

  def forward():
    return True

  def backward():
    return True

  def train():
    return True

  def test():
    return True


class Layer():
  def __init__(self, size, child_layer, init):
    self.size = size
    self.child_layer = child_layer
    # One bias for the layer ?
    self.bias = 1
    # edges between nodes below and my activations; initialize weights randomly between a given value from its - to its +
    self.weight_matrix = np.random.uniform(low=-init, high=init, size=(self.child_layer.size, self.size))
    # partial_deriv activations w.r.t. weights
    self.weight_derivs = np.zeros([self.child_layer.size, self.size])
    self.activations = np.ones(self.size)
    # nodes above w.r.t. each activation
    self.derivs = np.zeros(self.size)
    self.lr = 0.1

  def forward(self):
    """
    Forward propogate the activations from the last layer
    to this layer by combining previous activations, weight parameters, bias, and applying
    the nonlinear activation function.
    """
    print("WEIGHTS")
    print(self.weight_matrix)
    self.activations = self._vectorized_sigmoid(self.child_layer.activations.T.dot(self.weight_matrix) + self.bias)

  def backward(self, lr=0.1):
    """
    Backward propogate the error, updating the weights and bias of this layer
    """
    self.lr = lr
    # Compute partial of my nodes w.r.t. my weights
    # Use np.outer to compute the matrix from the product of the 2 activations
    self.weight_derivs = np.outer(self._vectorized_sigmoid(self.child_layer.activations), \
                         self._vectorized_derivative_sigmoid(self.activations)) \
                          * self.derivs * self.lr

    # Compute partial deriv of my nodes w.r.t. my child_layer's nodes
    self.child_layer.derivs = (self.derivs * self._vectorized_derivative_sigmoid(self.activations)).dot(self.weight_matrix.T)

    # Lastly, we can update the weights for this layer)
    self.weight_matrix -= self.weight_derivs

  def _sigmoid(self, x):
    return 1 / 1 + np.exp(-x)

  def _derivative_sigmoid(self, x):
    return self._sigmoid(x) * (1.0 - self._sigmoid(x))

  def _vectorized_sigmoid(self, X):
    """
    sigmoid for each scalar in a numpy vector
    """
    sig = np.vectorize(self._sigmoid)
    return sig(X)

  def _vectorized_derivative_sigmoid(self, X):
      """
      Derivative sigmoid for each scalar in a numpy vector
      """
      deriv_sig = np.vectorize(self._derivative_sigmoid)
      return deriv_sig(X)

  def _softmax(self, x):
    exponentials = [np.exp(p) for p in x]
    denominator = sum(exponentials)
    return [p / denominator for p in exponentials]


class InputLayer(Layer):
  def __init__(self, size):
    self.size = size
    # One bias for the layer
    self.bias = 1
    self.weight_matrix = None
    # Instantiate an np array with the input values
    self.activations = np.ones(self.size)

  def forward(self, inputs):
    self.activations = np.array(inputs)

  def backward(self):
    None


class LossLayer(Layer):
  """
  We expect this to just be one node.
  """
  def __init__(self, child_layer):
    self.loss = 0.0
    self.deriv_loss = 0.0
    self.child_layer = child_layer

  def _softmax(self, x):
    """
    Softmax function;

    x is a list: the output activations
    """
    exponentials = [np.exp(p) for p in x]
    denominator = sum(exponentials)
    return [p / denominator for p in exponentials]

  def _negative_log_likelihood(self, y_hat):
    """
    Used to compute the loss over the softmax layer
    """
    return -np.log(y_hat)

  def _deriv_negative_log_likelihood(self, y_hat):
    return -1/y_hat

  def forward(self, targets):
    """
    Compute the loss
    """
    # Make sure we have an NP array, so we can perform the vector math
    targets = np.array(targets)
    softmax = self._softmax(self.child_layer.activations)
    print("CONFIDENCE: %.2f" % targets.dot(softmax))
    # Because targets are 1 hot, this will just lookup the value of our softmax
    # (essentially the classifier's confidence) for the correct class
    self.loss = self._negative_log_likelihood(targets.dot(softmax))

  def backward(self, targets):
    """
    Compute the derivative of the loss
    """
    # Make sure we have an NP array, so we can perform the vector math
    targets = np.array(targets)
    softmax = self._softmax(self.child_layer.activations)

    self.child_layer.derivs = self._deriv_negative_log_likelihood(targets.dot(softmax)) * \
                              self._vectorized_sigmoid(self.child_layer.activations)
