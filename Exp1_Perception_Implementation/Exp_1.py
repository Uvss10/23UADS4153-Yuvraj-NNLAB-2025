import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # Update weights and bias
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.activation_function(linear_output)

# NAND Truth Table
X_nand = np.array([[0, 0],
                   [0, 1],
                   [1, 0],
                   [1, 1]])
y_nand = np.array([1, 1, 1, 0])  # NAND outputs

# XOR Truth Table
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([0, 1, 1, 0])  # XOR outputs

# Training and evaluating the Perceptron on NAND
print("Training on NAND:")
perceptron_nand = Perceptron(learning_rate=0.1, n_iter=10)
perceptron_nand.fit(X_nand, y_nand)
predictions_nand = perceptron_nand.predict(X_nand)
print("Predictions:", predictions_nand)
print("Actual:", y_nand)
print()

# Training and evaluating the Perceptron on XOR
print("Training on XOR:")
perceptron_xor = Perceptron(learning_rate=0.1, n_iter=10)
perceptron_xor.fit(X_xor, y_xor)
predictions_xor = perceptron_xor.predict(X_xor)
print("Predictions:", predictions_xor)
print("Actual:", y_xor)