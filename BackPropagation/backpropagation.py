import numpy as np

np.random.seed(42)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def bce_loss(y_pred, y_true):

    # Prevent log(0)
    y_pred = np.clip(y_pred, 1e-8, 1 - 1e-8)

    return -np.mean(
        y_true * np.log(y_pred) +
        (1 - y_true) * np.log(1 - y_pred)
    )


def forward(X, W1, b1, W2, b2):

    # Hidden layer
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)

    # Output layer
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "A2": A2
    }

    return A2, cache


def backward(Y, W2, cache):

    X = cache["X"]
    Z1 = cache["Z1"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    m = X.shape[1]

    # Output gradients
    dZ2 = A2 - Y
    dW2 = (1 / m) * (dZ2 @ A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    # Hidden gradients
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * sigmoid_derivative(Z1)

    dW1 = (1 / m) * (dZ1 @ X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dW1": dW1,
        "db1": db1,
        "dW2": dW2,
        "db2": db2
    }

    return grads


def update_parameters(W1, b1, W2, b2, grads, lr):

    W1 -= lr * grads["dW1"]
    b1 -= lr * grads["db1"]

    W2 -= lr * grads["dW2"]
    b2 -= lr * grads["db2"]

    return W1, b1, W2, b2


X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])

Y = np.array([
    [0, 1, 1, 0]
])


n_input = 2
n_hidden = 4
n_output = 1

lr = 0.5
epochs = 5000


# Random initialization
W1 = np.random.randn(n_hidden, n_input) * 0.5
b1 = np.zeros((n_hidden, 1))

W2 = np.random.randn(n_output, n_hidden) * 0.5
b2 = np.zeros((n_output, 1))


losses = []

for epoch in range(epochs):

    A2, cache = forward(X, W1, b1, W2, b2)

    loss = bce_loss(A2, Y)
    losses.append(loss)

    grads = backward(Y, W2, cache)

    W1, b1, W2, b2 = update_parameters(
        W1, b1, W2, b2, grads, lr
    )

    if (epoch + 1) % 1000 == 0:

        predictions = (A2 >= 0.5).astype(int)
        accuracy = np.mean(predictions == Y) * 100

        print(
            f"Epoch {epoch + 1:5d} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {accuracy:.0f}%"
        )


print("\nFinal Predictions:\n")

test_inputs = [
    ([0], [0]),
    ([0], [1]),
    ([1], [0]),
    ([1], [1])
]

true_outputs = [0, 1, 1, 0]

for i, (x1, x2) in enumerate(test_inputs):

    x = np.array([x1, x2])

    prediction, _ = forward(x, W1, b1, W2, b2)

    print(
        f"Input [{x1[0]}, {x2[0]}] "
        f"-> Predicted: {float(prediction):.3f} "
        f"| True: {true_outputs[i]}"
    )

print(f"\nInitial Loss : {losses[0]:.4f}")
print(f"Final Loss   : {losses[-1]:.4f}")