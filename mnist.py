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

    def activation(self, h):
        # sigmoid function
        return 1 / (1 + math.exp(-h))

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
        hidden_activations = np.zeros(self.hidden_units + 1)    # hidden activations +1 for bias
        output_activations = np.zeros(self.output_units)        # output activations

        for epoch in range(epochs):
            num_correct = 0.0

            # inner loop will run 60,000 times
            for (x, target) in zip(x_train, y_train):
                x = x.flatten()
                x = np.insert(x, 0, 1.0)        # bias

                # forward phase
                for i in range(self.hidden_units):
                    h = np.dot(x, self.hidden_weights[i])
                    hidden_activations[i + 1] = self.activation(h)   # activation of hidden layer neurons

                hidden_activations[0] = 1.0     # bias 

                for j in range(self.output_units):
                    h = np.dot(hidden_activations, self.output_weights[j])
                    output_activations[j] = self.activation(h)       # activation of output layer neurons

                prediction = np.argmax(output_activations)

                # backward phase
                if prediction == target:
                    # all fine and dandy
                    num_correct += 1
                else:
                    # not all fine and dandy
                    self.fix_weights()

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
    epochs = 50

    sys.exit(main(learning_rate, momentum, hidden_units, epochs))