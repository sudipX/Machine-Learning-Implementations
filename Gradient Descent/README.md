# Neural Network Optimizers from Scratch using NumPy

This project implements a small neural network from scratch using only NumPy and compares different optimization algorithms:

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- Adam Optimizer
- AdamW Optimizer

The goal of this implementation is educational:
understanding how neural network training and optimizers work internally without using deep learning frameworks like PyTorch or TensorFlow.

---

# Features

- Fully connected 2-layer neural network
- Binary classification on synthetic 2D dataset
- Forward propagation
- Backpropagation
- Binary Cross Entropy Loss
- Multiple optimizer implementations from scratch
- Accuracy evaluation
- Pure NumPy implementation

---

# Dataset

The dataset consists of two Gaussian clusters:

- Class 0 centered near `(0,0)`
- Class 1 centered near `(2,2)`

Generated using:

```python
X = np.vstack([
    np.random.randn(n//2,2) * 0.6,
    np.random.randn(n//2,2) * 0.6 + 2.0
])
```

This creates a simple linearly separable binary classification dataset.

---

# Neural Network Architecture

## Input Layer
- 2 features

## Hidden Layer
- 8 neurons
- Sigmoid activation

## Output Layer
- 1 neuron
- Sigmoid activation

---

# Loss Function

Binary Cross Entropy Loss:

```math
L = -\frac{1}{m}\sum \left[
Y \log(A) + (1-Y)\log(1-A)
\right]
```

---

# Implemented Optimizers

## 1. SGD

Basic gradient descent update:

```math
\theta = \theta - \eta \nabla_\theta
```

---

## 2. Momentum SGD

Uses moving average of gradients:

```math
v_t = \beta v_{t-1} + (1-\beta)g_t
```

```math
\theta = \theta - \eta v_t
```

Helps accelerate convergence and reduce oscillations.

---

## 3. Adam

Adaptive Moment Estimation.

Tracks:
- First moment (momentum)
- Second moment (variance)

Update rule:

```math
\theta = \theta - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
```

---

## 4. AdamW

Adam with decoupled weight decay regularization.

Update rule:

```math
\theta =
\theta -
\eta
\left(
\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
+
\lambda \theta
\right)
```

Used widely in modern deep learning systems and transformers.

---

# Project Structure

```text
gradient.py
README.md
```

---

# How to Run

Clone the repository:

```bash
git clone <your-repo-link>
cd <repo-name>
```

Run the script:

```bash
python gradient.py
```

---

# Example Output

```text
SGD (lr = 0.1)
[SGD     ] Epoch 100 | Train Loss: 0.23
...
[ADAMW   ] Final Test Accuracy : 97.5%
```

---

# Concepts Practiced

This project helps understand:

- Matrix dimensions in neural networks
- Forward propagation
- Backpropagation
- Gradient computation
- Momentum optimization
- Adaptive optimization
- Bias correction in Adam
- Weight decay regularization
- NumPy broadcasting
- Vectorized implementation

---


# Technologies Used

- Python
- NumPy

---

# Educational Purpose

This project is built for learning and research-oriented understanding of optimization algorithms in deep learning.

The implementation prioritizes mathematical clarity and transparency over framework abstraction.