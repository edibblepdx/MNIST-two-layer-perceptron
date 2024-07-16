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
        # sigmoid function
        return 1 / (1 + math.exp(-a))

    def output_error(self):
        a = 1

    def hidden_error(self):
        a = 1

    def fix_weights(self):
        a = 1

    def train(self, x_train, y_train, x_test, y_test, epochs):
        """
        for each input vector:
            forwards phase:
                - compute the activation of each neuron j in the hidden layer(s)
                - work through the network until you get to the output layer neurons,
                  which have activations
            backwards phase:
                - compute the error at the output
                - compute the error in the hidden layer(s)
                - update the output layer weights
                - update the hidden layer weights
            recall
        """
        train_accuracies = []
        test_accuracies = []
        output_activations = np.zeros(self.output_units)        # output activations
        hidden_activations = np.zeros(self.hidden_units + 1)    # hidden activations +1 for bias
        output_errors = np.zeros(self.output_units)
        hidden_errors = np.zeros(self.hidden_units)
        output_deltas = np.zeros(np.shape(self.output_weights))
        hidden_deltas = np.zeros(np.shape(self.hidden_weights))

        for epoch in range(epochs):
            # i-th input
            # j-th hidden unit
            # k-th output unit
            num_correct = 0.0

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
                else:
                    # not all fine and dandy
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

                    # output deltas
                    for k in range(self.output_units):
                        # k-th output unit
                        for j in range(self.hidden_units + 1):
                            # j-th hidden unit
                            output_deltas[k][j] = (
                                (self.learning_rate 
                                * output_errors[k] 
                                * hidden_activations[j]) 
                                + (self.momentum 
                                * output_deltas[k][j])
                            )

                    # hidden deltas
                    for j in range(self.hidden_units):
                        # j-th hidden unit
                        for i in range(self.input_size + 1):
                            # i-th input
                            hidden_deltas[j][i] = (
                                (self.learning_rate 
                                * hidden_errors[j]
                                * x[i])
                                + (self.momentum
                                * hidden_deltas[j][i])
                            )

                    # fix weights
                    self.output_weights = np.add(self.output_weights, output_deltas)
                    self.hidden_weights = np.add(self.hidden_weights, hidden_deltas)

            train_accuracy = num_correct / len(x_train)
            train_accuracies.append(train_accuracy)

            #test_accuracy = self.evaluate(x_test, y_test)
            #test_accuracies.append(test_accuracy)

            #print(f'Epoch {epoch} : Correct Train {num_correct:.0f} : Accuracy Train {train_accuracy:.4f} : Accuracy Test {test_accuracy:.4f}')
            print(f'Epoch {epoch} : Correct Train {num_correct:.0f} : Accuracy Train {train_accuracy:.4f}')

        return train_accuracies, test_accuracies
            

def main(learning_rate, momentum, hidden_units, epochs):
    # load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale data to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    perceptron = Perceptron(learning_rate, momentum, hidden_units)
    perceptron.train(x_train, y_train, x_test, y_test, epochs)

    return 0

if __name__ == '__main__':
    """
    weights = (-.05 < w < .05)

    Experiment 1:
        learning_rate = 0.1
        momentum = 0.9
        hidden_units = {20, 50, 100}
        epochs = 50

    Experiment 2:
        learning_rate = 0.1
        momentum = {0, 0.25, 0.50}
        hidden_units = 100
    
    Experiment 3: 
        Train two networks, using respectively one quarter and one half of the training
        examples for training. Make sure that in each case your training data is approximately
        balanced among the 10 different classes.

        hidden_units = 100
        momentum = 0.9

    Change weights after each training example and return:
        plot of both training and test accuracy as a function of epoch number 
        confusion matrix for each of your trained networks

        Each output unit corresponds to one of the 10 classes (0 to 9). Set the target
        value tk for output unit k to 0.9 if the input class is the kth class, 0.1 otherwise.
    """

    learning_rate = 0.1
    momentum = 0.9
    hidden_units = 20
    epochs = 5

    sys.exit(main(learning_rate, momentum, hidden_units, epochs))