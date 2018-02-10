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
      if i == 0
        # input is the child for the first hidden layer
        self.hidden_layers.append(Layer(self.hidden_size, self.input_layer))
      else:
        # Previous hidden layer is the child for each subsequent hidden layer
        self.hidden_layers.append(Layer(self.hidden_size, self.hidden_layers[i-1]))
    # Set the output layer with last hidden layer as its child
    self.output_layer = Layer(output_size, hidden_layers[-1])

    # instantiate blank LR, can be set in the train() method
    self.learning_rate = 0.0

  def forward():
    return True

  def backward():
    return True

  def train():
    return True

  def test():
    return True


class Layer():
  def __init__(size, child_layer):
    self.size = size
    self.child_layer = child_layer
    # One bias for the layer ?
    self.bias = 1
    # edges between nodes below and my activations
    self.weight_matrix = np.random.randn(self.child_layer.size, self.size)
    # partial_deriv activations w.r.t. weights
    self.weight_derivs = np.zeros(self.child_layer.size, self.size)
    self.activations = np.ones(self.size)
    # nodes above w.r.t. each activation
    self.derivs = np.zeros(self.size)

  def forward():
    """
    Forward propogate the activations from the last layer
    to this layer by combining previous activations, weight parameters, bias, and applying
    the nonlinear activation function.
    """
    self.activations = self._vectorized_sigmoid(self.child_layer.activations.dot(self.weight_matrix) + self.bias)

  def backward():
    """
    Backward propogate the error, updating the weights and bias of this layer
    """
    # Compute partial of my nodes w.r.t. my weights
    self.weight_derivs = self._vectorized_sigmoid(self.child_layer.activations) * self._vectorized_derivative_sigmoid(self.activations) \
                          * self.derivs * self.learning_rate

    # Compute partial of my nodes w.r.t. my child_layer's nodes
    ################
    # Not positive this dot product is quite right? Too tired to fix...
    self.child_layer.derivs = (self.derivs * self._vectorized_derivative_sigmoid(self.activations)).dot(self.weight_matrix.T)

  def _sigmoid(self, x):
    return np.exp(-x)

  def _derivative_sigmoid(self, x):
    return self._sigmoid(x) * (1.0 - self._sigmoid(x))

  def _vectorized_sigmoid(self, X):
    """
    sigmoid for each scalar in a numpy vector
    """
    sig = np.vectorize(self.sigmoid)
      return sig(X)

  def _vectorized_derivative_activation(self, X):
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
  def __init__(input_values):
    self.size = len(input_values)
    # One bias for the layer
    self.bias = 1
    self.weight_matrix = nil
    #TODO is this how you instantiate an ndarray with defined values?
    self.activations = np.ndarray(input_values)

  def forward():
    continue

  def backward():
    continue


class LossLayer():
  def __init__():

  def _negative_log_likelihood(self, y_hat):
    """
    Used to compute the loss over the softmax layer
    """
    return -np.log(y_hat)

  def _deriv_negative_log_likelihood(self, y_hat):
    return -1/y_hat
