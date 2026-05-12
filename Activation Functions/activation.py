import numpy as np


def summarise(name, z_vals, fn, dfn):

    print(f"\n{'=' * 50}")
    print(f"{name:^50}")
    print(f"{'=' * 50}")

    print(f"{'z':>10} | {'f(z)':>12} | {'f_prime(z)':>14}")
    print(f"{'-' * 42}")

    for z in z_vals:
        print(f"{z:>10.2f} | {fn(z):>12.4f} | {dfn(z):>14.4f}")


def step(z):
    return np.where(z >= 0, 1.0, 0.0)


def step_derivative(z):
    return np.zeros_like(z)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh_fn(z):
    return np.tanh(z)


def tanh_derivative(z):
    return 1.0 - np.tanh(z) ** 2


def relu(z):
    return np.maximum(0.0, z)


def relu_derivative(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1.0))


def elu_derivative(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(z))


def gelu(z):

    c = 0.7978845608028654

    return (
        0.5
        * z
        * (1.0 + np.tanh(c * (z + 0.044715 * z**3)))
    )


def gelu_derivative(z):

    h = 1e-5

    return (gelu(z + h) - gelu(z - h)) / (2.0 * h)


def softmax(z):

    z_stable = z - np.max(z)

    exp_z = np.exp(z_stable)

    return exp_z / np.sum(exp_z)


z_test = np.array([-3.0, -1.0, -0.78, 0.0, 1.0, 3.0])

summarise("SIGMOID", z_test, sigmoid, sigmoid_derivative)

summarise("TANH", z_test, tanh_fn, tanh_derivative)

summarise("RELU", z_test, relu, relu_derivative)

summarise(
    "LEAKY RELU (alpha = 0.01)",
    z_test,
    leaky_relu,
    leaky_relu_derivative
)

summarise(
    "ELU (alpha = 1.0)",
    z_test,
    elu,
    elu_derivative
)

summarise("GELU", z_test, gelu, gelu_derivative)


logits = np.array([2.0, 1.0, 0.1])

probs = softmax(logits)

print(f"\n{'=' * 50}")
print(f"{'SOFTMAX':^50}")
print(f"{'=' * 50}")

print(f"Logits          : {logits}")
print(f"Softmax Output  : {probs.round(4)}")
print(f"Sum of Probabilities : {probs.sum():.6f}")
print(f"Predicted Class : {np.argmax(probs)}")


print("\n=== VANISHING GRADIENT DEMO ===")

print("\nSigmoid Layers:")

z = 2.0

for n_layers in [1, 5, 10, 20, 50]:

    g = 1.0

    for _ in range(n_layers):
        g *= sigmoid_derivative(z)

    print(
        f"{n_layers:>2d} layers -> gradient = {g:.2e}"
    )


print("\nReLU Layers:")

z = 1.0

for n_layers in [1, 5, 10, 20, 50]:

    g = 1.0

    for _ in range(n_layers):
        g *= relu_derivative(z)

    print(
        f"{n_layers:>2d} layers -> gradient = {g:.2e}"
    )