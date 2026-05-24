# Activation Functions : NumPy Reference Implementation

A from-scratch NumPy implementation of the most commonly used neural network activation functions, along with their derivatives. Includes a softmax demo and a vanishing gradient demonstration.

---

## What It Covers

Each activation function is implemented alongside its analytical derivative (except GELU, which uses numerical differentiation). Running the script prints a formatted table of output values and gradients for a fixed set of test inputs, making it easy to compare function behaviour side by side.

The functions implemented are Step, Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, and GELU. Softmax is also included and demonstrated separately on a set of logits.

---

## Requirements

```bash
pip install numpy
```

---

## Running

```bash
python activation.py
```

The script prints three sections: activation tables for each function, the softmax output with probabilities and predicted class, and the vanishing gradient comparison between sigmoid and ReLU across layer depths.

---

## Functions

`sigmoid`, `tanh_fn`, `relu`, `leaky_relu`, `elu`, `gelu` : activation functions with their corresponding `_derivative` counterparts.

`step` : binary step function included for reference; its derivative is zero everywhere and is not suitable for gradient-based training.

`softmax` : numerically stable softmax using max subtraction before exponentiation.

`summarise` : utility that prints a formatted table of `f(z)` and `f'(z)` values for a given activation function over the test input array.
