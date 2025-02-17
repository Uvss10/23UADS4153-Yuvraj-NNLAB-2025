import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0  # Flatten and normalize
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0    # Flatten and normalize
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)   # One-hot encode labels
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)     # One-hot encode labels

# Define the neural network architecture
input_size = 28 * 28  # Input layer size (flattened MNIST images)
hidden_size = 128     # Hidden layer size
output_size = 10      # Output layer size (10 classes for digits 0-9)

# Initialize weights and biases
weights = {
    'hidden': tf.Variable(tf.random.normal([input_size, hidden_size])),
    'output': tf.Variable(tf.random.normal([hidden_size, output_size]))
}
biases = {
    'hidden': tf.Variable(tf.random.normal([hidden_size])),
    'output': tf.Variable(tf.random.normal([output_size]))
}

# Define the activation function (ReLU for hidden layer, softmax for output layer)
def relu(x):
    return tf.maximum(0, x)

def softmax(x):
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis=1, keepdims=True)

# Define the feed-forward function
def feed_forward(x):
    hidden_layer = relu(tf.matmul(x, weights['hidden']) + biases['hidden'])
    output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
    return output_layer

# Define the loss function (cross-entropy)
def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

# Define the optimizer (gradient descent)
learning_rate = 0.01
optimizer = tf.optimizers.SGD(learning_rate)

# Training the model
epochs = 10
batch_size = 128

for epoch in range(epochs):
    for i in range(0, len(x_train), batch_size):
        # Get a batch of data
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Perform forward pass
        with tf.GradientTape() as tape:
            y_pred = feed_forward(x_batch)
            loss = cross_entropy_loss(y_batch, y_pred)

        # Perform backpropagation
        gradients = tape.gradient(loss, list(weights.values()) + list(biases.values()))
        optimizer.apply_gradients(zip(gradients, list(weights.values()) + list(biases.values())))

    # Print loss for each epoch
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.numpy()}")

# Evaluate the model on the test set
y_pred_test = feed_forward(x_test)
predicted_labels = tf.argmax(y_pred_test, axis=1)
true_labels = tf.argmax(y_test, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, true_labels), tf.float32))
print(f"Test Accuracy: {accuracy.numpy() * 100:.2f}%")