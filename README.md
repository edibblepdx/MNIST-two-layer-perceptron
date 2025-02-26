# Results:
For each experiment, learning rate
was set to 0.1, epochs to 50, and used a batch size of 1. The default momentum was
0.9 and default hidden units was 100. Weights were initialized to small random numbers
(−0.5 < w < 0.5). Additionally, the inputs were randomly permuted for each epoch.

## Experiment 1: {20, 50, 100} hidden units
- Increasing trend in accuracy with more hidden units.
- Increased speed of convergence (flattening) with 50 and 100 hidden units.
- Increased overfitting with with 50 and 100 hidden units.

## Experiment 2: {0.00, 0.25, 0.50} momentum
- Light positive relationship with accuracy on the training set; accuracy on the test set remained stagnant.
- Significant positive relationship with the speed of convergence.
- Slightly increased overfitting with momentum.

## Experiment 3: {0.25, 0.50} fraction of training set
- More data implied higher accuracy.
- More data implied faster convergence.
- More data implied less overfitting.
