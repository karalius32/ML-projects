import numpy as np


class Sigmoid():
  def __init__(self):
    self.layer_type= "activation"

  def forward(self, X):
    self.output = 1 / (1 + np.exp(-X))
    return self.output

  def backward(self, gradient):
    self.gradient = self.output * (1 - self.output) * gradient
    return self.gradient


class ReLU():
  def __init__(self):
    self.layer_type= "activation"

  def forward(self, X):
    self.input = X
    self.output = np.clip(np.where(X > 0, X, 0), 0, 2)
    return self.output

  def backward(self, gradient):
    self.gradient = np.where(self.input > 0, 1, 0) * gradient
    return self.gradient


class Linear():
  def __init__(self):
    self.layer_type= "activation"

  def forward(self, X):
    self.input = X
    return self.input

  def backward(self, gradient):
    self.gradient = np.ones(self.input.shape) * gradient
    return self.gradient


class SoftMax():
  def __init__(self):
    self.layer_type = "activation_softmax"

  def forward(self, X):
    self.inputs = X
    exp_X = np.exp(X)
    self.output = exp_X / np.sum(exp_X, axis=1).reshape(-1, 1)
    return self.output

  def backward(self, gradient, y_true): 
    batch_size = len(self.inputs)
    l = len(self.inputs[0])
    derivatives = np.zeros((batch_size, l))
    for i in range(batch_size):
      k = y_true[i]
      for j in range(l):
        if j == k:
          derivatives[i, j] = self.output[i, j] * (1 - self.output[i, k])
        else:
          derivatives[i, j] = -self.output[i, j] * self.output[i, k]
    self.gradient = derivatives * gradient.reshape(-1, 1)
    return self.gradient


class Layer():
  def __init__(self, input_size, layer_size, initialization_technique="randn_rand"):
    if initialization_technique == "rand_zeros":
      self.W = np.random.rand(input_size, layer_size)
      self.b = np.zeros((1, layer_size))
    elif initialization_technique == "randn_rand":
      self.W = np.random.randn(input_size, layer_size)
      self.b = np.random.rand(1, layer_size)
    elif initialization_technique == "randn_zeros":
      self.W = np.random.randn(input_size, layer_size)
      self.b = np.zeros(layer_size)
    self.layer_type = "layer"
    self.W_momentums = np.zeros_like(self.W)
    self.b_momentums = np.zeros_like(self.b)
    self.iteration = 0

  def forward(self, X):
    self.input = X
    self.output = np.matmul(X, self.W) + self.b
    return self.output

  def backward(self, gradient):
    self.dW = np.matmul(self.input.T, gradient)
    self.db = np.sum(gradient, axis=0)
    self.gradient = np.matmul(gradient, self.W.T)
    return self.gradient

  def optimize(self, learning_rate, lr_decay=0, momentum=0):
    self.iteration += 1
    current_learning_rate = learning_rate / (1 + lr_decay * self.iteration)
    W_updates = self.W_momentums * momentum - self.dW * current_learning_rate
    b_updates = self.b_momentums * momentum - self.db * current_learning_rate
    self.W_momentums = W_updates
    self.b_momentums = b_updates
    self.W += W_updates
    self.b += b_updates

class MSE():
  def __init__(self):
    pass

  def forward(self, y_pred, y_true):
    self.error = y_pred - y_true
    self.output = np.mean(self.error ** 2)
    return self.output

  def backward(self):
    return self.error


class CategoricallCrossEntropy():
  def __init__(self):
    pass

  def forward(self, y_pred, y_true):
    self.y_pred = y_pred
    self.y_true = y_true
    self.losses = np.array([-np.log(np.clip(y_p[y_t], 1e-9, 1)) for y_t, y_p in zip(self.y_true, y_pred)])
    return np.mean(self.losses)

  def backward(self):
    return np.array([-1/y_p[y_t] for y_t, y_p in zip(self.y_true, self.y_pred)])


class Model_Base():
  def __init__(self, sequential):
    self.sequential = sequential
    self.history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

  def predict(self, X):
    for layer in self.sequential:
      X = layer.forward(X)
    return X

  def backward(self, gradient, y_true=None):
    for layer in reversed(self.sequential):
      if layer.layer_type == "activation_softmax":
        gradient = layer.backward(gradient, y_true)
      else:
        gradient = layer.backward(gradient)

  def optimize(self, learning_rate):
    for layer in self.sequential:
      if layer.layer_type == "layer":
        layer.optimize(learning_rate)

  def fit(self, X, y, epochs, learning_rate, loss_fn, batch_size, val_data=None):
    # Training
    for i in range(epochs):
      indeces = np.random.choice(len(X), len(X), replace=False)
      for j in range(len(X) // batch_size):
        X_batch = X[indeces[j * batch_size : j * batch_size + 10]]
        y_batch = y[indeces[j * batch_size : j * batch_size + 10]]
        y_pred = self.predict(X_batch)
        loss_fn.forward(y_pred, y_batch)
        gradient = loss_fn.backward()
        self.backward(gradient, y_batch)
        self.optimize(learning_rate)
      # Saving and printing info
        
      y_pred_train = self.predict(X)
      loss_train = loss_fn.forward(y_pred_train, y)
      acc_train = accuracy(y_pred_train, y)
      self.history["train_loss"].append(loss_train)
      self.history["train_accuracy"].append(acc_train)
      '''
      if type(val_data) != type(None):
        X_val, y_val = val_data
        y_pred_val = self.predict(X_val)
        loss_val = loss_fn.forward(y_pred_val, y_val)
        acc_val = accuracy(y_pred_val, y_val)
        self.history["val_loss"].append(loss_val)
        self.history["val_accuracy"].append(acc_val)
      '''
      

def accuracy(y_pred, y_true):
  if len(y_true.shape) == 1:
    return np.sum(np.argmax(y_pred, axis=1) == y_true) / len(y_pred)
  return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1)) / len(y_pred)

class Model(Model_Base):
  def __init__(self, layers, output_dropdown):
    self.sequential = []
    for i in range(1, len(layers.layers)):
        self.sequential.append(Layer(layers.layers[i - 1].n, layers.layers[i].n))
        if layers.layers[i].buttons != None:
          if layers.layers[i].activation_name == "ReLU":
            self.sequential.append(ReLU())
          elif layers.layers[i].activation_name == "Sigmoid":
            self.sequential.append(Sigmoid())
          elif layers.layers[i].activation_name == "Linear":
            self.sequential.append(Linear())
        else:
          self.sequential.append(SoftMax())
    if output_dropdown.getSelected() == None or output_dropdown.getSelected() == "Softmax":
      self.sequential[-1] = SoftMax()
    elif output_dropdown.getSelected() == "Sigmoid":
      self.sequential[-1] = Sigmoid()
    elif output_dropdown.getSelected() == "Linear":
      self.sequential[-1] = Linear()
    super().__init__(self.sequential)

  def get_weights_and_biases(self):
    weights = []
    biases = [[1, 1]]
    for layer in self.sequential:
      if layer.layer_type == "layer":
        weights.append(layer.W)
        biases.append(layer.b[0])
    return weights, biases

'''
if layers.layers[i].buttons != None:
  if layers.layers[i].buttons["activation"] == None or layers.layers[i].buttons["activation"].getSelected() == "ReLU":
    self.sequential.append(ReLU())
  elif layers.layers[i].buttons["activation"].getSelected() == "Sigmoid":
    self.sequential.append(Sigmoid())
'''
