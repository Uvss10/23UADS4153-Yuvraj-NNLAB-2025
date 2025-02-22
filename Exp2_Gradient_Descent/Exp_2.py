import numpy as np

# Step Activation Function
def step_function(x):
    return np.where(x >= 0, 1, 0)

# XOR Dataset
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])  # Expected output

# Network Parameters
input_size = 2
hidden_size = 2  # Two hidden neurons
output_size = 1
lr = 0.1  # Learning rate
epochs = 10000  # Training iterations

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size)  # Input to hidden layer
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)  # Hidden to output layer
b2 = np.zeros((1, output_size))

# Training loop (Manual Update Without Gradient)
for epoch in range(epochs):
    # Forward Pass
    hidden_input = np.dot(X, W1) + b1
    hidden_output = step_function(hidden_input)  # Apply Step Function

    final_input = np.dot(hidden_output, W2) + b2
    final_output = step_function(final_input)  # Step Activation for output

    # Compute error
    error = y - final_output

    # Manual Weight Update (since Step function is non-differentiable)
    W2 += hidden_output.T.dot(error) * lr
    b2 += np.sum(error, axis=0, keepdims=True) * lr
    W1 += X.T.dot(error.dot(W2.T)) * lr
    b1 += np.sum(error.dot(W2.T), axis=0, keepdims=True) * lr

    # Print loss every 1000 epochs
    if epoch % 1000 == 0:
        loss = np.mean(error**2)
        print(f"Epoch {epoch}, Loss: {loss:.5f}")

# Testing the trained model
print("\nFinal Predictions:")
for i in range(len(X)):
    hidden_output = step_function(np.dot(X[i], W1) + b1)
    output = step_function(np.dot(hidden_output, W2) + b2)
    print(f"Input: {X[i]}, Predicted Output: {output[0]}, Expected: {y[i][0]}")
