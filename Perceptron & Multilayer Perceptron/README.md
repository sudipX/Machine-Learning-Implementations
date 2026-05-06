# Multi-Layer Perceptron from Scratch using NumPy

## Overview

This project implements a Multi-Layer Perceptron (MLP) from scratch using only NumPy. No deep learning frameworks such as PyTorch, TensorFlow, or scikit-learn are used. The goal is to understand how neural networks work internally by implementing forward propagation, loss computation, and backpropagation manually.

The model performs binary classification on a simple synthetic dataset with two Gaussian-distributed classes.

---

## Dataset

We generate a simple 2D dataset:

- Class 0: points centered around (0, 0)
- Class 1: points centered around (2, 2)

Each class contains 100 samples, resulting in a total of 200 data points.

\[
X \in \mathbb{R}^{200 \times 2}, \quad y \in \{0,1\}^{200 \times 1}
\]

The dataset is shuffled and split into:
- 80% training set
- 20% test set

---

## Model Architecture

The neural network consists of:

- Input layer: 2 features
- Hidden layer: 4 neurons (sigmoid activation)
- Output layer: 1 neuron (sigmoid activation)

Structure:

2 → 4 → 1

---

## Forward Propagation

Let:

- \(A_0 = X^T\)
- \(W_1, b_1\): parameters of hidden layer
- \(W_2, b_2\): parameters of output layer

### Hidden Layer

\[
Z_1 = W_1 A_0 + b_1
\]

\[
A_1 = \sigma(Z_1)
\]

### Output Layer

\[
Z_2 = W_2 A_1 + b_2
\]

\[
A_2 = \sigma(Z_2)
\]

Sigmoid function:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

---

## Loss Function

Binary cross-entropy loss:

\[
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log(a_2^{(i)}) + (1 - y^{(i)}) \log(1 - a_2^{(i)}) \right]
\]

To avoid numerical instability, predictions are clipped:

\[
A_2 \in [10^{-8}, 1 - 10^{-8}]
\]

---

## Backpropagation

### Output Layer

\[
\frac{\partial L}{\partial Z_2} = A_2 - Y
\]

\[
dW_2 = \frac{1}{m} dZ_2 A_1^T
\]

\[
db_2 = \frac{1}{m} \sum dZ_2
\]

---

### Hidden Layer

\[
dA_1 = W_2^T dZ_2
\]

\[
dZ_1 = dA_1 \cdot \sigma'(Z_1)
\]

Where:

\[
\sigma'(Z_1) = \sigma(Z_1)(1 - \sigma(Z_1))
\]

\[
dW_1 = \frac{1}{m} dZ_1 A_0^T
\]

\[
db_1 = \frac{1}{m} \sum dZ_1
\]

---

## Parameter Update

Gradient descent update rule:

\[
W = W - \alpha dW
\]

\[
b = b - \alpha db
\]

Where:
- \( \alpha \) = learning rate

---

## Training Procedure

For each epoch:

1. Forward propagation
2. Compute loss
3. Backpropagation
4. Update weights using gradient descent

Loss is recorded at every epoch to monitor training progress.

---

## Evaluation

Prediction is done using:

- Forward pass on test data
- Thresholding at 0.5

Accuracy is computed as:

\[
Accuracy = \frac{\text{Correct Predictions}}{\text{Total Samples}} \times 100
\]

---

## Key Insights

- Small weight initialization prevents sigmoid saturation
- Backpropagation is a direct application of the chain rule
- Bias gradients are summed across all samples
- Weight gradients depend on both error and input activations
- Matrix multiplication naturally aggregates gradients over the dataset

---

## Requirements

- Python 3.x
- NumPy

---

## Summary

This project demonstrates how a neural network learns from scratch without using any deep learning framework. Every component, from forward propagation to gradient descent, is implemented manually using NumPy.