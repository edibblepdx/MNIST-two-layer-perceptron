# Ethan Dibble
# 
# MNIST two-layer perceptron
# 
# Change the parameters below the name guard at the bottom of this file.
# There you will also find a description and values for the experiments ran.
# Learning rate was set to 0.1 and epochs to 50 for each experiment performing
# stochastic gradient descent with a batch size of one.

import sys
import math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

INPUT_SIZE = 784
OUTPUT_UNITS = 10
WEIGHT_SCALE = 0.5

class Perceptron:
    def __init__(self, learning_rate, momentum, hidden_units, output_units=OUTPUT_UNITS, input_size=INPUT_SIZE):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.hidden_units = hidden_units 
        self.output_units = output_units
        self.input_size = input_size
        self.hidden_weights = WEIGHT_SCALE * np.random.randn(hidden_units, input_size + 1)
        self.output_weights = WEIGHT_SCALE * np.random.randn(output_units, hidden_units + 1)

    def activation(self, a):
        """
        Activation sigmoid function
        """
        return 1 / (1 + math.exp(-a))

    def forward(self, output_activations, hidden_activations, x):
        """
        Forward phase
        """
        # hidden activations
        for j in range(self.hidden_units):
            # j-th hidden unit
            a = np.dot(x, self.hidden_weights[j])
            hidden_activations[j + 1] = self.activation(a)   # activation of hidden layer neurons

        hidden_activations[0] = 1.0     # bias 

        # output activations
        for k in range(self.output_units):
            # k-th output unit
            a = np.dot(hidden_activations, self.output_weights[k])
            output_activations[k] = self.activation(a)       # activation of 

    def errors(self, output_activations, output_errors, hidden_activations, hidden_errors, target): 
        """
        Get errors
        """
        # output errors
        for k in range(self.output_units):
            t = 0.9 if k == target else 0.1
            y = output_activations[k]
            # be careful of sign
            output_errors[k] = y * (1 - y) * (t - y)

        # hidden errors
        for j in range(self.hidden_units):
            h = hidden_activations[j + 1]   # bias is index 0
            # be careful of sign
            sum = 0.0
            for k in range(self.output_units):
                sum += self.output_weights[k][j] * output_errors[k]
            hidden_errors[j] = h * (1 - h) * sum

    def weight_updates(self, output_errors, output_weight_updates, hidden_activations, hidden_errors, hidden_weight_updates, x):
        """
        Get weight weight updates
        """
        # output weight updates
        for k in range(self.output_units):
            # k-th output unit
            for j in range(self.hidden_units + 1):
                # j-th hidden unit
                output_weight_updates[k][j] = (
                    (self.learning_rate 
                    * output_errors[k] 
                    * hidden_activations[j]) 
                    + (self.momentum 
                    * output_weight_updates[k][j])
                )

        # hidden weight updates
        for j in range(self.hidden_units):
            # j-th hidden unit
            for i in range(self.input_size + 1):
                # i-th input
                hidden_weight_updates[j][i] = (
                    (self.learning_rate 
                    * hidden_errors[j]
                    * x[i])
                    + (self.momentum
                    * hidden_weight_updates[j][i])
                )
    
    def permute(self, x_train, y_train):
        """
        Create copies of x_train and y_train with unified permutations
        """
        assert len(x_train) == len(y_train)
        rng = np.random.default_rng()
        p = rng.permutation(len(x_train))
        return x_train[p], y_train[p]


    def train(self, x_train, y_train, x_test, y_test, epochs):
        """
        Train the model
        """
        train_accuracies = []                                               # train accuracies
        test_accuracies = []                                                # test accuracies
        output_activations = np.zeros(self.output_units)                    # output activations
        hidden_activations = np.zeros(self.hidden_units + 1)                # hidden activations +1 for bias
        output_errors = np.zeros(self.output_units)                         # output node errors
        hidden_errors = np.zeros(self.hidden_units)                         # hidden node errors
        output_weight_updates = np.zeros(np.shape(self.output_weights))     # output weight updates
        hidden_weight_updates = np.zeros(np.shape(self.hidden_weights))     # hidden weight updates

        for epoch in range(epochs + 1):
            # i-th input
            # j-th hidden unit
            # k-th output unit
            num_correct = 0.0       # counter for accuracy on the train set

            # permute x_train and y_train
            x_train, y_train = self.permute(x_train, y_train)

            # inner loop will run 60,000 times
            for (x, target) in zip(x_train, y_train):
                x = x.flatten()
                x = np.insert(x, 0, 1.0)        # bias

                # forward phase
                # hidden activations
                for j in range(self.hidden_units):
                    # j-th hidden unit
                    a = np.dot(x, self.hidden_weights[j])
                    hidden_activations[j + 1] = self.activation(a)   # activation of hidden layer neurons

                hidden_activations[0] = 1.0     # bias 

                # output activations
                for k in range(self.output_units):
                    # k-th output unit
                    a = np.dot(hidden_activations, self.output_weights[k])
                    output_activations[k] = self.activation(a)       # activation of output layer neurons

                # predict
                prediction = np.argmax(output_activations)

                # backward phase
                if prediction == target:
                    # all fine and dandy
                    num_correct += 1

                elif epoch != 0:
                    # not all fine and dandy
                    
                    # errors
                    self.errors(output_activations, output_errors, hidden_activations, hidden_errors, target)

                    # weight updates
                    self.weight_updates(output_errors, output_weight_updates, hidden_activations, hidden_errors, hidden_weight_updates, x)

                    # fix weights
                    self.output_weights = np.add(self.output_weights, output_weight_updates)
                    self.hidden_weights = np.add(self.hidden_weights, hidden_weight_updates)

            # train accuracy
            train_accuracy = num_correct / len(x_train)
            train_accuracies.append(train_accuracy)

            # test accuracy
            test_accuracy = self.evaluate(x_test, y_test)
            test_accuracies.append(test_accuracy)

            print(f'Epoch {epoch} : Correct Train {num_correct:.0f} : Accuracy Train {train_accuracy:.4f} : Accuracy Test {test_accuracy:.4f}')

        return train_accuracies, test_accuracies
            
    def evaluate(self, x_test, y_test):
        """
        Evaluate model on the test set
        """
        output_activations = np.zeros(self.output_units)        # output activations
        hidden_activations = np.zeros(self.hidden_units + 1)    # hidden activations +1 for bias
        num_correct = 0.0                                       # counter for accuracy on the test set

        for (x, target) in zip(x_test, y_test):
            x = x.flatten()
            x = np.insert(x, 0, 1.0)  # Adding bias input
            self.forward(output_activations, hidden_activations, x)

            # predict
            prediction = np.argmax(output_activations)

            if prediction == target:
                num_correct += 1

        return num_correct / len(x_test)

    def confusion_matrix(self, x_test, y_test):
        """
        Create a confusion matrix on the test set
        """
        output_activations = np.zeros(self.output_units)        # output activations
        hidden_activations = np.zeros(self.hidden_units + 1)    # hidden activations +1 for bias
        y_true = []     # true value
        y_pred = []     # prediction

        for (x, target) in zip(x_test, y_test):
            x = x.flatten()
            x = np.insert(x, 0, 1.0)  # Adding bias input
            self.forward(output_activations, hidden_activations, x)

            # predict
            prediction = np.argmax(output_activations)

            y_true.append(target)
            y_pred.append(prediction)

        return confusion_matrix(y_true, y_pred)

def train_subset(x_train, y_train, fraction, num_classes=10):
    """
    take only a balanced subset of the training set
    """
    # ensure a balanced training set
    samples_per_class = int((len(x_train) * fraction) / num_classes)   

    x_subset = []   # subset of inputs
    y_subset = []   # subset of labels

    for c in range(num_classes):
        # get the indexes
        i = np.where(y_train == c)[0]   # np.where returns a tuple
        i = np.random.choice(i, size=samples_per_class, replace=False)

        # append those elements to the subsets
        x_subset.append(x_train[i])
        y_subset.append(y_train[i])

    # have to use np.concatenate instead of np.array
    return np.concatenate(x_subset, axis=0), np.concatenate(y_subset, axis=0)

def main(learning_rate, momentum, hidden_units, epochs, fraction):
    # load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # take a subset of the data
    if fraction != 1:
        x_train, y_train = train_subset(x_train, y_train, fraction)

    # training shapes
    print("x_train: ", np.shape(x_train))
    print("y_train: ", np.shape(y_train))

    # Scale data to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Train
    perceptron = Perceptron(learning_rate, momentum, hidden_units)
    train_accuracies, test_accuracies = perceptron.train(x_train, y_train, x_test, y_test, epochs)

    # Plot
    max_ticks = 10 # max ticks on x axis
    step = max(1, len(train_accuracies) // max_ticks)
    ticks = range(0, len(train_accuracies), step)

    plt.plot(range(len(train_accuracies)), train_accuracies, label='train')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='test')
    plt.ylim(0, 1)
    plt.xticks(ticks, ticks)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(f'Perceptron Learning Accuracy (η={learning_rate})')
    plt.legend(loc='lower right')
    plt.show()

    # Confusion matrices for the perceptron on the test set
    cm = perceptron.confusion_matrix(x_test, y_test)
    display = ConfusionMatrixDisplay(confusion_matrix=cm)
    display.plot()
    plt.title(f'Perceptron Confusion Matrix on Test Set (η={learning_rate})')
    plt.show()

    return 0

if __name__ == '__main__':
    """
    learning_rate = 0.1
    initial weights = (-.05 < w < .05)
    batch size = 1
    epochs = 50

    Experiment 1:
        momentum = 0.9
        hidden_units = {20, 50, 100}
        fraction = 1

    Experiment 2:
        momentum = {0, 0.25, 0.50}
        hidden_units = 100
        fraction = 1
    
    Experiment 3: 
        momentum = 0.9
        hidden_units = 100
        fraction = {0.25, 0.5}
    """

    learning_rate = 0.1
    momentum = 0.9
    hidden_units = 20
    epochs = 1
    fraction = 1            # what % of the test set to use

    sys.exit(main(learning_rate, momentum, hidden_units, epochs, fraction))