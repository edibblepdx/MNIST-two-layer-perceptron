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
            output_activations[k] = self.activation(a)       # activation of 

    def errors(self, output_activations, output_errors, hidden_activations, hidden_errors, target): 
        """
        Get errors
        """
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
        train_accuracies = []
        test_accuracies = []
        output_activations = np.zeros(self.output_units)        # output activations
        hidden_activations = np.zeros(self.hidden_units + 1)    # hidden activations +1 for bias
        output_errors = np.zeros(self.output_units)
        hidden_errors = np.zeros(self.hidden_units)
        output_weight_updates = np.zeros(np.shape(self.output_weights))
        hidden_weight_updates = np.zeros(np.shape(self.hidden_weights))

        for epoch in range(epochs + 1):
            # i-th input
            # j-th hidden unit
            # k-th output unit
            num_correct = 0.0

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
        num_correct = 0.0

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
        y_true = []
        y_pred = []

        for (x, target) in zip(x_test, y_test):
            x = x.flatten()
            x = np.insert(x, 0, 1.0)  # Adding bias input
            self.forward(output_activations, hidden_activations, x)

            # predict
            prediction = np.argmax(output_activations)

            y_true.append(target)
            y_pred.append(prediction)

        return confusion_matrix(y_true, y_pred)


def main(learning_rate, momentum, hidden_units, epochs):
    # load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

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
    weights = (-.05 < w < .05)
    batch size = 1
    epochs = 50

    Experiment 1:
        learning_rate = 0.1
        momentum = 0.9
        hidden_units = {20, 50, 100}

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
    """

    learning_rate = 0.1
    momentum = 0.9
    hidden_units = 20
    epochs = 1

    sys.exit(main(learning_rate, momentum, hidden_units, epochs))