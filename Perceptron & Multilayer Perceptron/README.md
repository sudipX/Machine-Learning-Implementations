# Multi-Layer Perceptron from Scratch in NumPy

This repository contains a comprehensive implementation of a Multi-Layer Perceptron (MLP) built entirely from the ground up using **NumPy**. By avoiding high-level frameworks like PyTorch or TensorFlow, this project provides a transparent look at the underlying linear algebra and calculus that drive neural network learning.

## Project Architecture

The model is designed for binary classification with an architecture optimized for clarity:
*   **Input Layer:** 2 features ($x_1, x_2$).
*   **Hidden Layer:** 4 neurons with Sigmoid activation.
*   **Output Layer:** 1 neuron with Sigmoid activation.
*   **Optimization:** Batch Gradient Descent with Binary Cross-Entropy loss.

---

## Mathematical Derivations

### 1. Forward Propagation
Forward propagation is the process of transforming input data into a prediction through sequential matrix operations.

For any layer $l$, the pre-activation $Z^{[l]}$ is calculated as the weighted sum of the previous layer's output plus a bias:
$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$

The activation $A^{[l]}$ is then computed by applying the Sigmoid function $\sigma$ element-wise:
$$A^{[l]} = \sigma(Z^{[l]}) = \frac{1}{1 + e^{-Z^{[l]}}}$$

### 2. Loss Function
To measure the discrepancy between the predicted probability $A^{[2]}$ and the actual label $Y$, we use the Binary Cross-Entropy loss function. For $m$ samples:
$$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a^{(i)}) + (1 - y^{(i)}) \log(1 - a^{(i)}) \right]$$

### 3. Backward Propagation
Backpropagation calculates the gradient of the loss function with respect to each parameter using the chain rule.

**Step 1: Output Layer Gradients**
The derivative of the loss with respect to the output pre-activation $Z^{[2]}$ is derived from the combination of Sigmoid and Cross-Entropy, which simplifies elegantly to:
$$dZ^{[2]} = \frac{\partial L}{\partial Z^{[2]}} = A^{[2]} - Y$$

The gradients for the weights and biases of the output layer are:
$$dW^{[2]} = \frac{1}{m} dZ^{[2]} \cdot (A^{[1]})^T$$
$$db^{[2]} = \frac{1}{m} \sum dZ^{[2]}$$

**Step 2: Hidden Layer Gradients**
To find the gradient for the hidden layer, we first compute how the loss changes with respect to the hidden layer activation $A^{[1]}$:
$$dA^{[1]} = (W^{[2]})^T \cdot dZ^{[2]}$$

We then apply the derivative of the Sigmoid function, where $\sigma'(Z) = \sigma(Z)(1 - \sigma(Z))$, to find the error signal at the hidden layer using element-wise multiplication ($\odot$):
$$dZ^{[1]} = dA^{[1]} \odot \sigma'(Z^{[1]})$$

Finally, we calculate the gradients for the first layer parameters:
$$dW^{[1]} = \frac{1}{m} dZ^{[1]} \cdot (A^{[0]})^T$$
$$db^{[1]} = \frac{1}{m} \sum dZ^{[1]}$$

---

## Implementation Overview

### Activation Function
The project utilizes the Sigmoid function for all neurons. This maps any real-valued input into a range between 0 and 1, making it ideal for probability-based classification.

### Initialization
Weights are initialized using a small random scale (0.01) from a normal distribution. This prevents "Saturation", where large initial values force the Sigmoid function into flat regions (near 0 or 1) where the gradient is nearly zero, effectively stopping the learning process before it begins.

---

## Execution and Results

### Usage
To execute the training and evaluation, ensure you have NumPy installed:
```bash
pip install numpy
python perceptron.py