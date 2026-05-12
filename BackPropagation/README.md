# XOR Neural Network : From Scratch with NumPy

A minimal two-layer neural network built from scratch using only NumPy, trained to solve the XOR problem. No ML frameworks are used, every component including forward propagation, backpropagation, and gradient descent is implemented manually.

---

## What It Does

The XOR problem is a classic test for neural networks because it is not linearly separable : a single-layer perceptron cannot solve it. This script trains a small feedforward network to correctly learn the XOR truth table:

| Input A | Input B | Output |
|---------|---------|--------|
| 0       | 0       | 0      |
| 0       | 1       | 1      |
| 1       | 0       | 1      |
| 1       | 1       | 0      |

---

## Network Architecture

- **Input layer** : 2 neurons (one per input bit)
- **Hidden layer** : 4 neurons with sigmoid activation
- **Output layer** : 1 neuron with sigmoid activation
- **Loss function** : Binary Cross-Entropy (BCE)
- **Optimizer** : Vanilla gradient descent

---

## How It Works

The script manually implements each step of the training loop. On each epoch, it runs a forward pass to compute predictions, calculates the BCE loss, runs backpropagation to compute gradients for all weights and biases, and updates the parameters using gradient descent.

Training runs for 5000 epochs with a learning rate of 0.5. Loss and accuracy are printed every 1000 epochs. After training, the script tests all four XOR input combinations and prints the predicted vs. true outputs alongside the initial and final loss.

---

## Requirements

```bash
pip install numpy
```

---

## Running

```bash
python xor_nn.py
```

Sample output:

```
Epoch  1000 | Loss: 0.3012 | Accuracy: 75%
Epoch  2000 | Loss: 0.1894 | Accuracy: 100%
...
Final Predictions:

Input [0, 0] -> Predicted: 0.023 | True: 0
Input [0, 1] -> Predicted: 0.971 | True: 1
Input [1, 0] -> Predicted: 0.972 | True: 1
Input [1, 1] -> Predicted: 0.031 | True: 0
```

---

## Functions

`sigmoid(z)` and `sigmoid_derivative(z)` : activation function and its gradient used during backpropagation.

`bce_loss(y_pred, y_true)` : computes binary cross-entropy loss with numerical clipping to avoid log(0).

`forward(X, W1, b1, W2, b2)` : runs a forward pass through both layers and caches intermediate values needed for backprop.

`backward(Y, W2, cache)` : computes gradients for all weights and biases using the chain rule.

`update_parameters(...)` : applies gradient descent to update all parameters.
