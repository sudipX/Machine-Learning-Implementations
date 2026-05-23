import numpy as np


# Step Function
def step(z):
    return np.where(z>=0,1.0,0.0)

def step_derivative(z):
    return np.zeros_like(z)


# Sigmoid Function
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s*(1.0-s)


# Tanh Function
def tanh_fn(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1.0 - np.tanh(z)**2


# ReLU Function
def relu(z):
    return np.maximum(0.0,z)

def relu_derivative(z):
    return (z>0).astype(float)


# Leaky ReLU Function
def leaky_relu(z, alpha=0.01):
    return np.where(z>0, z, alpha*z)

def leaky_relu_derivative(z,alpha=0.01):
    return np.where(z>0,1.0,alpha)


# Exponential Linear Unit (ELU)
def elu(z,alpha=0.01):
    return np.where(z>0,z,alpha*(np.exp(z)-1.0))

def elu_derivative(z,alpha=1.0):
    return np.where(z>0,1.0,alpha*np.exp(z))


# GELU (Gaussian Error Linear Unit)
def gelu(z):
    c =  0.7978845608028654 # sqrt(2/pi) precomputed for speed
    return 0.5*z*(1+np.tanh(c*(z+0.044415 * z**3)))

def gelu_derivative(z):
    h = 1e-5
    return (gelu(z+h) - gelu(z-h)) / (2.0 * h)



# Softmax function

def softmax(z):
    z_stable = z - np.max(z) #The Numerical Stability Fix so that no overflow occurs
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z)

z_test = np.array([-3.0, -1.0, -0.78, 0.0, 1.0, 3.0])

print(f"Step:\n {step(z_test)}\n {step_derivative(z_test)}\n ")
print(f"Sigmoid: \n {sigmoid(z_test)}\n{sigmoid_derivative(z_test)}\n ")
print(f"ReLU:\n {relu(z_test)}\n{relu_derivative(z_test)}\n ")
print(f"Leaky ReLU: {leaky_relu(z_test)}\n{leaky_relu_derivative(z_test)}\n ")
print(f"GELU: {gelu(z_test)}\n{gelu_derivative(z_test)}\n ")
print(f"Softmax: {softmax(z_test)}\n") 

