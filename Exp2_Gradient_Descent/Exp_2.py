import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=1000):
        self.weights = np.random.randn(input_size + 1)  # +1 for bias
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation(self, x):
        return 1 if x >= 0 else 0

    def train(self, X, y):
        X = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias column
        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                y_pred = self.activation(np.dot(self.weights, X[i]))
                self.weights += self.learning_rate * (y[i] - y_pred) * X[i]

    def predict(self, X):
        y_pred = []
        for x in X:
            x_with_bias = np.insert(x, 0, 1)  # Add bias term
            prediction = self.activation(np.dot(self.weights, x_with_bias))
            y_pred.append(prediction)
        return y_pred
# Define functions dynamically
Hidden_ly_output = [
    np.array([0, 0, 0, 1]),
    np.array([0, 0, 1, 0]),
    np.array([0, 1, 0, 0]),
    np.array([1, 0, 0, 0])
]

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = []

hidden_perceptrons = []  # List to store trained perceptrons


# Train perceptrons for each function dynamically i.e 4 neurons for 4 different inputs
for y in Hidden_ly_output:
    perceptron = Perceptron(input_size=2,epochs=25)
    perceptron.train(X, y)
    y_pred = perceptron.predict(X)
    predictions.append(y_pred)
    hidden_perceptrons.append(perceptron)
 
# Convert predictions into input for final perceptron
final_X = np.array(predictions)

final_y = np.array([0, 1, 1, 0]) # XOR output


# Train final perceptron
final_perceptron = Perceptron(input_size=len(final_X),epochs=25)
final_perceptron.train(final_X, final_y)

final_predictions = final_perceptron.predict(final_X)

# Display XOR truth table with predictions
print("\nXOR Truth Table Predictions:")
print(" X1  X2 |  y_actual  y_pred")
print("---------------------------")
for i in range(len(X)):
    print(f" {X[i][0]}   {X[i][1]}  |     {final_y[i]}        {final_predictions[i]}")

import random
#Input
input_data = np.array([random.randrange(0, 2),random.randrange(0, 2)])

# Step 1: Get hidden layer outputs
hidden_outputs = []
for p in hidden_perceptrons:
    hidden_output = p.predict([input_data])
    hidden_outputs.append(hidden_output)

# Step 2: Convert hidden outputs to NumPy array for final perceptron
hidden_outputs = np.array([hidden_outputs])

# Step 3: Get final prediction
final_prediction = final_perceptron.predict(hidden_outputs)[0]

# Print result
print(f"Predicted output for {input_data}: {final_prediction}")


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score


accuracy = accuracy_score(final_y, final_predictions)
print(f"Final Perceptron Accuracy: {accuracy * 100:.2f}%")
print()

cm = confusion_matrix(final_y, final_predictions)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix for XOR using MLP")
plt.show()
print()


