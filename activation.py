import numpy as np

def sigmoid(x):
    x = 1./(1 + np.exp(-x))
    return x

def sigmoid_grad(x):
    da = sigmoid(x) * ( 1 - sigmoid(x))
    return da

def relu(x):
    x = np.where(x > 0, x, 0)
    return x

def relu_grad(x):
    da = np.where(x > 0, 1, 0)
    return da

def leaky_relu(x):
  x = np.where(x > 0, x, 0.2 * x)
  return x

def leaky_relu_grad(x):
  da = np.where(x > 0, 1, 0.2)
  return da

def tanh(x):
    x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return x

def tanh_grad(x):
    da =  1 - tanh(x) ** 2
    return da

def softmax(x):
    z_max = np.amax(x,axis =  1, keepdims = True)
    z_max = np.exp(x - z_max)
    probs= np.sum(z_max,axis = 1, keepdims = True)
    probs = z_max / probs 
    return probs