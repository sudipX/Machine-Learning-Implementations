import numpy as np


# Mean Squared Error
def mse_loss (y_true, y_pred):
    n = len(y_true)
    residuals = y_true - y_pred
    return np.mean(residuals**2)


def mse_gradient(y_true, y_pred):
    n = len(y_true)
    return 2.0 * (y_pred-y_true) / n


# Mean Absolute Error
def mae_loss (y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mae_gradient (y_true, y_pred):
    n = len(y_true)
    return np.sign(y_pred - y_true) / n


# Binary Cross Entropy
def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    per_sample = -(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return np.mean(per_sample)

def bce_gradient(y_true, y_pred):
    n = len(y_true)
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return (-(y_true/y_pred) + (1-y_true)/(1-y_pred)) / n

def bce_sigmoid_gradient(y_true, y_pred):
    return (y_pred - y_true) / len(y_true)

# Categorical Cross Entropy

def cce_loss(y_true_onehot, y_pred_softmax):
    # Here both have Size : (n,k), n = number of samples and k = number of categorical class
    y_pred_softmax = np.clip(y_pred_softmax, 1e-7,1.0)

    per_sample = -np.sum(y_true_onehot * np.log(y_pred_softmax),axis=1)

    return np.mean(per_sample)

def cce_softmax_gradient(y_true_onehot, y_pred_softmax):
    n = y_true_onehot.shape[0]
    return (y_pred_softmax - y_true_onehot) / n


#Huber Loss
def huber_loss(y_true, y_pred, delta = 1.0):
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)

    quadratic = 0.5 * residuals ** 2
    linear = delta * (abs_residuals - 0.5*delta)

    loss_per_sample = np.where(abs_residuals <=delta, quadratic, linear)

    return np.mean(loss_per_sample)

def huber_gradient (y_true,y_pred,delta=1.0):

    n = len(y_true)
    residuals = y_pred - y_true
    abs_residuals = np.abs(residuals)

    grad_quad = residuals
    grad_lin = delta * np.sign(residuals)

    return np.where(abs_residuals <= delta, grad_quad, grad_lin) / n

y_reg = np.array([3.0, 5.0, 2.0, 8.0]) # true values
yh_reg = np.array([2.5, 5.5 , 3.9, 5.0]) # predictions

print ( f" True : { y_reg }")
print ( f" Predicted : { yh_reg }")
print ( f" Residuals : { y_reg - yh_reg }")
print ( f"\ nMSE = { mse_loss (y_reg , yh_reg ):.4f}")
print ( f"MAE = { mae_loss (y_reg , yh_reg ):.4f}")
print ( f" Huber = { huber_loss (y_reg , yh_reg , delta =1.0) :.4f}")
print ( f"\ nMSE gradient : { mse_gradient (y_reg , yh_reg ). round (4)}")
print ( f"MAE gradient : { mae_gradient (y_reg , yh_reg ). round (4)}")
print ( f" Huber gradient : { huber_gradient (y_reg , yh_reg ). round (4)}")

print ("\n BINARY CROSS - ENTROPY \n")

y_bin = np . array ([1 , 0 , 1 , 0] , dtype = float ) # true labels
yh_bin = np . array ([0.9 , 0.1 , 0.4 , 0.8]) # predicted probs
print ( f" True labels : { y_bin }")
print ( f" Predictions : { yh_bin }")
print ( f"\nBCE loss = { bce_loss (y_bin , yh_bin ):.4f}")
print ( f"Per - sample BCE:")
for i in range (4) :
    yi , pi = y_bin [ i ] , yh_bin [ i ]
    li = -( yi * np . log ( pi +1e-7) + (1 - yi ) * np . log (1 - pi +1e-7) )
    status = " correct " if ( pi > 0.5) == ( yi == 1) else " WRONG "
    print ( f" Sample {i +1}: y={ yi :.0f} , p={ pi} , loss ={ li :.3f} [{status }]")


print ("\n CATEGORICAL CROSS - ENTROPY \n")


# 3 samples , 3 classes
y_cat = np . array ([[1 ,0 ,0] ,[0 ,1 ,0] ,[0 ,0 ,1]] , dtype = float ) # one -hot
yh_cat = np . array ([[0.7 ,0.2 ,0.1] ,[0.1 ,0.8 ,0.1] ,[0.2 ,0.3 ,0.5]])
print ( f"CCE loss = { cce_loss (y_cat , yh_cat ):.4f}")
for i in range (3) :
    true_class = np . argmax ( y_cat [ i ])
    pred_prob = yh_cat [i , true_class ]
    li = - np . log ( pred_prob + 1e-7)
    print ( f" Sample {i +1}: true class ={ true_class } , "f" predicted prob ={ pred_prob :.1f} , loss ={ li :.3f}")
        
