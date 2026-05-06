import numpy as np

np.random.seed(42)

#total number of data points
n_samples = 200

#Class 1 : 100 points centered around (0,0)
X0 = np.random.randn(n_samples // 2, 2) * 0.5

#Class 2 : 100 points centered around (2,2)
X1 = np.random.randn(n_samples // 2, 2) *0.5 + 2

X = np.vstack([X0,X1]) # shape (200,2)

y = np.vstack([
    np.zeros((n_samples// 2,1)), #labels for Class 1
    np.ones((n_samples // 2, 1)) #labels for Class 2
])

indices = np.random.permutation(n_samples)
X = X[indices]
y = y[indices]

split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]


#Activation Function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s *(1-s)


# Architecture : 2 inputs -> 4 hidden neurons -> 1 output
input_dim = 2 # x and y coordinates
hidden_dim = 4
output_dim = 1 # binary classification

#0.01 is multiplied to the weights to prevent sigmoid from saturating and stopping from learning.

# for hidden layer
W1 = np.random.randn(hidden_dim,input_dim) * 0.01 
b1 = np.zeros((hidden_dim,1))

#for output layer
W2 = np.random.randn(output_dim,hidden_dim) * 0.01
b2 = np.zeros((output_dim,1))

learning_rate = 0.1
epochs = 1000

loss_history = []

for epoch in range (epochs):

    A0 = X_train.T
    Z1 = W1@A0 + b1 #hidden layer preactivation
    A1 = sigmoid(Z1) #hidden layer activation

    Z2 = W2@A1 + b2 #output layer preactivation
    A2 = sigmoid(Z2) #output layer activation

    #loss computation
    a2_clip = np.clip(A2, 1e-8, 1- 1e-8) #for numerical stability

    Y = y_train.T
    
    #cross entropy for each sample and then taking its mean
    loss = -np.mean(Y*np.log(a2_clip)+(1-Y)*np.log(1-a2_clip))
    loss_history.append(loss)

    m = X_train.shape[0]

    #predicted - actual (dL/dZ2)
    dZ2 = A2 - Y

    #  dL/dW2 = (dL/dZ2) * (dZ2/dW2) = dZ2 * A1 (because Z2 = W2.A1+b2)
    dW2 = (1.0 / m) * dZ2 @ A1.T
    #Apply chain rule here. And sum is done to average out.
    db2 = (1.0/m) * np.sum(dZ2, axis=1, keepdims=True)

    #Core idea of hidden layer gradient. It is just chain rule.
    dA1 = W2.T @ dZ2
    #here is also chain rule.
    dZ1 = dA1 * sigmoid_derivative(Z1)

    #  dL/dW1 = (dL/dZ1) * (dZ1/dW1) = dZ1 * A0 (because Z2 = W1.A0+b1)
    dW1 = (1.0 / m) * dZ1 @ A0.T
    db1 = (1.0 / m) * np.sum(dZ1, axis = 1, keepdims=True)

    #updating the weights and biases in reducing direction of loss
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2

    if(epoch + 1)%100==0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss:.4f}")


def predict (X, W1, b1, W2, b2, threshold=0.5):

    A0 = X.T
    Z1 = W1 @ A0 + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)

    return (A2 >=threshold).astype(int)


y_pred = predict(X_test, W1, b1, W2, b2)

correct = np.sum(y_pred.flatten()==y_test.flatten())

accuracy = correct/len(y_test) * 100

print(f"\nTest Accuarcy : {accuracy:.1f}%")
print(f"Final training loss : {loss_history[-1]:.4f}")