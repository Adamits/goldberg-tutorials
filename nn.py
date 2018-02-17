import numpy as np

class Layer():
  def __init__(self, size, child_layer, init, add_bias=True):
    self.size = size
    self.child_layer = child_layer
    self.add_bias = add_bias

    # edges between nodes below and my activations; initialize weights randomly between a given value from its - to its +
    self.weight_matrix = np.random.uniform(low=-init, high=init, size=(self.child_layer.size, self.size))
    if self.child_layer.add_bias:
      self.weight_bias = np.random.uniform(low=-init, high=init, size=(self.size))

    # partial_deriv activations w.r.t. weights
    # Note +1 to child size for bias
    self.weight_derivs = np.zeros([self.child_layer.size, self.size])
    if self.child_layer.add_bias:
      self.weight_bias_derivs = np.zeros([self.size])

    self.activations = np.zeros(self.size)
    self.sigmoid_activations = np.zeros(self.size)

    # nodes above w.r.t. each activation
    self.derivs = np.zeros(self.size)
    # Set here to be overridden when .forward() is called
    self.lr = 0.0

  def forward(self):
    """
    Forward propogate the activations from the last layer
    to this layer by combining previous activations, weight parameters, bias, and applying
    the nonlinear activation function to each of those products.

    print("ACTIVATIONS")
    print(self.activations)
    print("WEIGHTS")
    print(self.weight_matrix)
    if self.child_layer.add_bias:
      print("BIAS WEIGHTS")
      print(self.weight_bias)
    """

    # Compute activations with dot product of activations below and connecting weights
    self.activations = self.child_layer.sigmoid_activations.T.dot(self.weight_matrix)

    # Add bias
    if self.child_layer.add_bias:
      self.activations += self.weight_bias

    self.sigmoid_activations = self._vectorized_sigmoid(self.activations)

  def backward(self, lr=0.9):
    """
    Backward propogate the error, updating the weights and bias of this layer
    """
    self.lr = lr

    # Compute L' w.r.t Node', to be multiplied by child activations
    derivative_loss_wrt_derivative_activations = self.derivs * self._vectorized_derivative_sigmoid(self.activations)
    # Compute partial deriv of my nodes w.r.t. my weights
    # Use np.outer to compute the matrix from the product of the 2 activations
    self.weight_derivs = np.outer(self.child_layer.sigmoid_activations, \
                          derivative_loss_wrt_derivative_activations) \
                           * self.lr

    if self.child_layer.add_bias:
      self.weight_bias_derivs = derivative_loss_wrt_derivative_activations * self.lr

    # Compute partial deriv of my nodes w.r.t. my child_layer's nodes
    self.child_layer.derivs = derivative_loss_wrt_derivative_activations.dot(self.weight_matrix.T)

    # Lastly, we can update the weights for this layer)
    self.weight_matrix -= self.weight_derivs
    if self.child_layer.add_bias:
      self.weight_bias -= self.weight_bias_derivs

  def _sigmoid(self, x):
    #return np.tanh(x)
    return 1 / (1 + np.exp(-x))

  def _derivative_sigmoid(self, x):
    #return 1 - np.tanh(x)**2
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


class InputLayer(Layer):
  def __init__(self, size, add_bias=True):
    self.size = size
    self.weight_matrix = None
    self.add_bias = add_bias
    # Instantiate an np array with the input values
    self.activations = np.ones(self.size)

  def forward(self, inputs):
    """
    Forward propogate the input layer
    """
    self.activations = np.array(inputs)

    #############################################################
    ## Because we should not be taking nonlinearity
    ## this will just be the same as the activations
    ## We store this because the layer above does not
    ## know whether its child is an input layer or a hidden layer
    self.sigmoid_activations = np.array(inputs)

  def backward(self):
    """
    Override the parent Layer class with so the input layer does nothing
    """
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
    Numerically Stable Softmax function.

    x is a list: the output activations
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

  def _binary_cross_ent(self, y_hat, y):
    return (-y * np.log(y_hat)) - ((1 - y) * np.log(1 - y_hat))

  def _deriv_binary_cross_ent(self, y_hat, y):
    return y_hat - y

  def _negative_log_likelihood(self, y_hat):
    """
    Used to compute the loss over the softmax layer
    """
    return -np.log(y_hat)

  def _deriv_negative_log_likelihood(self, y_hat):
    """
    Derivative of neg log likelihood, when given a softmax confidence
    is just -1 / predicted probability
    """
    return -1/y_hat

  def forward(self, targets):
    """
    Compute the loss
    """
    # Make sure we have an NP array, so we can perform the vector math
    self.loss = self._binary_cross_ent(self.child_layer.sigmoid_activations, targets)

  def backward(self, targets):
    """
    Compute the derivative of the loss, and Loss' w.r.t. the output layer
    """
    self.deriv_loss = self._deriv_binary_cross_ent(self.child_layer.sigmoid_activations, targets)

    self.child_layer.derivs = self.deriv_loss * self.child_layer.sigmoid_activations
