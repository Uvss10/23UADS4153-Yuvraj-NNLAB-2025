import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, n_iter=10000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
        # Initialize weights
        self.W1 = np.random.rand(input_size, hidden_size)  # Weights for input to hidden layer
        self.b1 = np.zeros((1, hidden_size))               # Bias for hidden layer
        self.W2 = np.random.rand(hidden_size, output_size) # Weights for hidden to output layer
        self.b2 = np.zeros((1, output_size))               # Bias for output layer

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y):
        # Calculate the error
        output_error = y - self.a2
        output_delta = output_error * self.sigmoid_derivative(self.a2)

        # Calculate the error for the hidden layer
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.a1)

        # Update weights and biases
        self.W2 += self.a1.T.dot(output_delta) * self.learning_rate
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.W1 += X.T.dot(hidden_delta) * self.learning_rate
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

    def fit(self, X, y):
        for _ in range(self.n_iter):
            self.forward(X)
            self.backward(X, y)

    def predict(self, X):
        output = self.forward(X)
        return np.round(output)  # Return binary output

# XOR Truth Table
X_xor = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
y_xor = np.array([[0],
                  [1],
                  [1],
                  [0]])  # XOR outputs

# Create and train the MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.1, n_iter=10000)
mlp.fit(X_xor, y_xor)

# Make predictions
predictions = mlp.predict(X_xor)

# Display results
print("Predictions:")
print(predictions)
print("Actual:")
print(y_xor)