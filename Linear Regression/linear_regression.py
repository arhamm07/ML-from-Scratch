import numpy as np


class LinearRegression:

    def __init__(self , lr = 0.001 , n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self ,X , y):
        # We need to initialize the weights and biases in order run gradient descent

        n_samples , n_features = X.shape
        # the weights should be zero for all number of features
        self.weights = np.zeros(n_features)
        # the bias should be zero
        self.bias = 0

        # now lets run gradient descent

        for _ in range(self.n_iters):
             y_predict = np.dot(X , self.weights) + self.bias

             # now lets calculate the gradients for the weights and biases
             dw = (1/n_samples) * np.dot(X.T , (y_predict - y))
             db = (1/n_samples) * np.sum(y_predict - y)

             # now lets update the weights and biases

             self.weights -= self.lr * dw
             self.bias -= self.lr * db

    def predict(self , X):
        y_predict = np.dot(X , self.weights) + self.bias
        return y_predict
