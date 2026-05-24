# Loss Functions from Scratch using NumPy

A simple educational implementation of commonly used Machine Learning and Deep Learning loss functions using only NumPy.

This project demonstrates:

- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Binary Cross Entropy (BCE)
- Categorical Cross Entropy (CCE)
- Huber Loss
- Gradients of each loss function

The goal of this project is to deeply understand:

- How loss functions work mathematically
- How gradients are computed
- Why certain losses are used for regression vs classification
- How backpropagation starts from the loss layer

---

# Features

- Pure NumPy implementation
- Beginner-friendly code
- Includes gradient derivations
- Numerical stability using `np.clip()`
- Regression + Classification losses
- Example demonstrations with outputs

---

# Implemented Loss Functions

# 1. Mean Squared Error (MSE)

Used mainly for regression problems.

### Formula

```math
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
```

### Gradient

```math
\frac{\partial L}{\partial \hat{y}} = \frac{2(\hat{y} - y)}{n}
```

---

# 2. Mean Absolute Error (MAE)

Measures absolute differences between prediction and ground truth.

### Formula

```math
MAE = \frac{1}{n} \sum_{i=1}^{n}|y_i - \hat{y}_i|
```

### Gradient

```math
\frac{\partial L}{\partial \hat{y}} = \frac{sign(\hat{y} - y)}{n}
```

---

# 3. Huber Loss

Huber Loss combines the advantages of MSE and MAE.

- Quadratic for small errors
- Linear for large errors
- Robust to outliers

### Formula

```math
L_\delta(a) =
\begin{cases}
\frac{1}{2}a^2 & |a| \le \delta \\
\delta(|a| - \frac{1}{2}\delta) & |a| > \delta
\end{cases}
```

---

# 4. Binary Cross Entropy (BCE)

Used for binary classification tasks.

### Formula

```math
BCE = -\left[y\log(p) + (1-y)\log(1-p)\right]
```

### Gradient

```math
\frac{\partial L}{\partial \hat{y}}
=
-\frac{y}{\hat{y}}
+
\frac{1-y}{1-\hat{y}}
```

### Simplified Sigmoid + BCE Gradient

```math
\frac{\partial L}{\partial z} = \hat{y} - y
```

---

# 5. Categorical Cross Entropy (CCE)

Used for multi-class classification problems.

### Formula

```math
CCE = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)
```

### Simplified Softmax + CCE Gradient

```math
\frac{\partial L}{\partial z} = \hat{y} - y
```

---

# Project Structure

```bash
.
├── loss_functions.py
└── README.md
```

---

# Requirements

Install NumPy:

```bash
pip install numpy
```

---

# Running the Project

Run the script:

```bash
python loss.py
```

---

# Example Output

## Regression Losses

```python
True : [3. 5. 2. 8.]
Predicted : [2.5 5.5 3.9 5. ]

Residuals : [ 0.5 -0.5 -1.9  3. ]

MSE = 3.2775
MAE = 1.4750
Huber = 1.1138
```

---

## Binary Cross Entropy

```python
True labels : [1. 0. 1. 0.]
Predictions : [0.9 0.1 0.4 0.8]

BCE loss = 0.8574
```

---

## Categorical Cross Entropy

```python
CCE loss = 0.4243
```

---

# Learning Goals

This project is useful for understanding:

- Deep Learning fundamentals
- Backpropagation
- Gradient computation
- Optimization
- Numerical stability
- Neural Network internals

Especially useful if you are implementing:

- Neural Networks from scratch
- Backpropagation manually
- Custom ML frameworks
- Research-oriented ML systems

---
