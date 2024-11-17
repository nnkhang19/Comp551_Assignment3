import numpy as np
from activation import *

class HiddenLayer:
    def __init__(self, num_neurons, activation, reg = 0.0, reg_type = 'l2'):
        assert activation in ["sigmoid", "relu", "leaky_relu", "tanh", "softmax"], "Activation must be in [sigmoid, relu, tanh, softmax]"
        assert reg_type in [None, 'l1', 'l2'], "regulaization must be in [None, l1, l2]"
        self.num_neurons = num_neurons
        self.W = None
        self.activation = activation
        self.reg = reg
        self.reg_type = reg_type

    def forward(self, X):
        if self.W is None:
            W_shape = (X.shape[1], self.num_neurons)
            self.W = np.random.normal(0, np.sqrt(2/(X.shape[1]+self.num_neurons)), W_shape)
        activation_function = eval(self.activation)
        self.Z = X @ self.W
        A = activation_function(self.Z)
        return A

    def backward(self, X, delta_prev):
        activation_grad_function = eval(self.activation + "_grad")
        z = self.Z
        delta = delta_prev * activation_grad_function(z)
        W_grad = X.T @ delta

        if self.reg_type == 'l1':
          W_grad += (np.sign(self.W) * self.reg) / len(X)
          return W_grad, delta

        if self.reg_type == 'l2':
          W_grad += (self.W * self.reg) / len(X)
          return W_grad, delta
        return W_grad, delta

    def __str__(self):
        return f"Number of neurons: {self.num_neurons}, Activation: {self.activation}"

class MLP:
    def __init__(self, learning_rate, num_class=2, reg = 1e-5, reg_type = 'l2'):
        self.layers = []
        self.reg = reg
        self.reg_type = reg_type
        self.num_class = num_class
        self.learning_rate = learning_rate

    def add_layer(self, num_neurons, activation):
        assert activation in ["sigmoid", "relu", "leaky_relu", "tanh", "softmax"], "Activation must be in [sigmoid, relu, tanh, softmax]"
        self.layers.append(HiddenLayer(num_neurons, activation, self.reg, self.reg_type))

    def forward(self, X):
        all_X = [X]
        for i in range(len(self.layers)):
          X = self.layers[i].forward(X)
          all_X.append(X)
        return all_X

    def compute_loss(self, Y, Y_hat):
        m = len(Y)
        correct_log_probs = np.sum(Y * np.log(Y_hat), axis = 1)
        data_loss = -np.mean(correct_log_probs) 
        reg_loss = 0.0
        if self.reg_type == 'l1':
            reg_loss = np.sum(self.reg * np.abs(self.layers[-1].W)) / m
        elif self.reg_type == 'l2':
            reg_loss = np.sum(self.reg * self.layers[-1].W) / m
        else:
            reg_loss = 0
        data_loss += reg_loss
        return data_loss

    def compute_delta_grad_last(self, Y, all_X):
        m = Y.shape[0]
        delta_last = (all_X[-1] - Y) / m
        grad_last = all_X[-2].T @ delta_last + (self.layers[-1].W * self.reg) / m
        return delta_last, grad_last

    def backward(self, Y, all_X):
        delta_prev, grad_last = self.compute_delta_grad_last(Y, all_X)

        grad_list = [grad_last]
        for i in range(len(self.layers) - 1)[::-1]:
            prev_layer = self.layers[i+1] # previous layer as backward direction
            layer = self.layers[i]
            X = all_X[i]
            delta_prev = delta_prev @ prev_layer.W.T # dot product of delta_prev and W_prev
            grad_W, delta_prev = layer.backward(X, delta_prev)
            grad_list.append(grad_W)
        grad_list = grad_list[::-1]
        return grad_list

    def update_weight(self, grad_list):
        for i, layer in enumerate(self.layers):
            grad = grad_list[i]
            layer.W = layer.W - self.learning_rate * grad

    def update_weight_momentum(self, grad_list, momentum_rate):
        if not hasattr(self, "momentum"):
            self.momentum = [np.zeros_like(grad) for grad in grad_list]
        v = 0.0
        for i, layer in enumerate(self.layers):
          grad = grad_list[i]
          v = momentum_rate * v + self.learning_rate * grad
          layer.W -= v

    def predict(self, X_test):
        Y_hat = self.forward(X_test)[-1]
        return Y_hat

    def __str__(self):
        reps = ""
        for i, layer in enumerate(self.layers):
            reps += f"Layer: {i} | {layer}\n"
        return reps
        