import sys
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

INPUT_SIZE = 784
OUTPUT_UNITS = 10
WEIGHT_SCALE = 0.5

class Perceptron:
    def __init__(self, learning_rate, momentum, hidden_units, epochs):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.hidden_units = hidden_units 
        self.epochs = epochs
        self.hidden_weights = WEIGHT_SCALE * np.random.randn(hidden_units, INPUT_SIZE + 1)
        self.output_weights = WEIGHT_SCALE * np.random.randn(OUTPUT_UNITS, hidden_units + 1)

    #def activation:

def main(learning_rate, momentum, hidden_units, epochs):
    # load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Scale data to be between 0 and 1
    x_train, x_test = x_train / 255.0, x_test / 255.0

    perceptron = Perceptron(learning_rate, momentum, hidden_units, epochs)

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

    Change weights after each training example and return:
        plot of both training and test accuracy as a function of epoch number 
        confusion matrix for each of your trained networks
    """

    learning_rate = 0.1
    momentum = 0.9
    hidden_units = 20
    epochs = 50

    sys.exit(main(learning_rate, momentum, hidden_units, epochs))