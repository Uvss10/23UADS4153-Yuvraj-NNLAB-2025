import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.reshape(-1, 28 * 28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype(np.float32) / 255.0

# One-hot encode the labels
num_classes = 10
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]

# Define hyperparameters
input_size = 28 * 28  # 784
hidden_size = 128      # Number of neurons in the hidden layer
output_size = num_classes  # 10 classes
learning_rate = 0.01
num_epochs = 10
batch_size = 64

# Initialize weights and biases
W1 = tf.Variable(tf.random.normal([input_size, hidden_size]))
b1 = tf.Variable(tf.random.normal([hidden_size]))
W2 = tf.Variable(tf.random.normal([hidden_size, output_size]))
b2 = tf.Variable(tf.random.normal([output_size]))

# Define the feed-forward operation
@tf.function
def feed_forward(X):
    layer_1 = tf.add(tf.matmul(X, W1), b1)
    layer_1 = tf.nn.relu(layer_1)  # Activation function
    out_layer = tf.add(tf.matmul(layer_1, W2), b2)
    return out_layer

# Define the loss function (cross-entropy)
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

# Define the optimizer
optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

# Define accuracy
def compute_accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training the model
for epoch in range(num_epochs):
    num_batches = int(x_train.shape[0] / batch_size)
    for i in range(num_batches):
        batch_x = x_train[i * batch_size:(i + 1) * batch_size]
        batch_y = y_train[i * batch_size:(i + 1) * batch_size]

        with tf.GradientTape() as tape:
            logits = feed_forward(batch_x)
            loss = compute_loss(logits, batch_y)

        gradients = tape.gradient(loss, [W1, b1, W2, b2])
        optimizer.apply_gradients(zip(gradients, [W1, b1, W2, b2]))

    # Calculate loss and accuracy on the training set
    train_logits = feed_forward(x_train)
    train_loss = compute_loss(train_logits, y_train)
    train_accuracy = compute_accuracy(train_logits, y_train)
    print(f'Epoch {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')

# Evaluate the model on the test set
test_logits = feed_forward(x_test)
test_accuracy = compute_accuracy(test_logits, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')
