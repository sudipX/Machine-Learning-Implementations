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
- Includes gradient derivations in code form
- Numerical stability using `np.clip()`
- Regression + Classification losses
- Sample demonstrations with outputs

---

# Implemented Loss Functions

## Regression Losses

### 1. Mean Squared Error (MSE)

Used mainly for regression problems.

Formula:

\[
MSE = \frac{1}{n} \sum (y_{true} - y_{pred})^2
\]

Gradient:

\[
\frac{2(y_{pred} - y_{true})}{n}
\]

---

### 2. Mean Absolute Error (MAE)

Measures absolute differences.

Formula:

\[
MAE = \frac{1}{n} \sum |y_{true} - y_{pred}|
\]

Gradient:

\[
\frac{sign(y_{pred} - y_{true})}{n}
\]

---

### 3. Huber Loss

Combines advantages of MSE and MAE.

- Quadratic for small errors
- Linear for large errors
- More robust to outliers

Formula:

\[
L_\delta(a) =
\begin{cases}
\frac{1}{2}a^2 & |a| \le \delta \\
\delta(|a| - \frac{1}{2}\delta) & otherwise
\end{cases}
\]

---

# Classification Losses

## 4. Binary Cross Entropy (BCE)

Used for binary classification.

Formula:

\[
BCE = -[y \log(p) + (1-y)\log(1-p)]
\]

Implemented with:

- Stable clipping
- Standard BCE gradient
- Simplified sigmoid + BCE gradient

---

## 5. Categorical Cross Entropy (CCE)

Used for multi-class classification.

Formula:

\[
CCE = -\sum y_i \log(\hat{y}_i)
\]

Includes:

- One-hot encoded labels
- Softmax probabilities
- Simplified Softmax + CCE gradient

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
python loss_functions.py
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




