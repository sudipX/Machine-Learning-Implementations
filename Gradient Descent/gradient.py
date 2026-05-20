import numpy as np
np.random.seed(42)

n = 200

X = np.vstack([
    np.random.randn(n//2,2) * 0.6,
    np.random.randn(n//2,2) * 0.6 + 2.0
])

Y = np.array([0]*(n//2) + [1]*(n//2),dtype = float).reshape(-1,1) #-1 : Find rows automatically and 1: number ofcolumns

idx = np.random.permutation(n)
X,Y = X[idx], Y[idx]
X_tr, Y_tr = X[:160].T, Y[:160].T
X_te, Y_te = X[160:].T, Y[160:].T

def init_params():
    np.random.seed(0)
    return{
        'W1' : np.random.randn(8,2)*0.1,
        'b1' : np.zeros((8,1)),
        'W2' : np.random.randn(1,8)*0.1,
        'b2' : np.zeros((1,1))
    }

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def forward_backward(params, X,Y):

    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']

    m = X.shape[1]

    Z1 = W1@X + b1
    A1 = sigmoid(Z1)
    Z2 = W2@A1 + b2
    A2 = sigmoid(Z2)

    eps = 1e-8
    loss = -np.mean(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))

    dZ2 = A2 - Y
    dW2 = (1/m) * dZ2 @ A1.T
    db2 = (1/m) * np.sum(dZ2, axis = 1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * A1 * (1-A1)
    dW1 = (1/m) * dZ1 @ X.T
    db1 = (1/m) * np.sum(dZ1, axis =1, keepdims = True)

    grads = {'W1':dW1, 'b1':db1, 'W2' : dW2, 'b2':db2}
    return loss, grads



#Stochastic Gradient Descent
def sgd_update(params, grads, lr):

    new_params = {}
    for key in params:
        new_params[key] = params[key] - lr*grads[key]
        
    return new_params



#SGD with momentum
def momentum_update(params, grads, velocity, beta1, lr):
    new_params = {}
    new_velocity = {}

    for key in params:
        new_velocity[key] = beta1 * velocity[key] + (1-beta1)*grads[key]
        new_params[key] = params[key] - lr*new_velocity[key]
    
    return new_params, new_velocity


#Adapative Moment Estimation (Adam) Optimizer
def adam_update(params, grads, m, v, t, lr, beta1, beta2, eps):
    new_params ={}
    new_m = {}
    new_v={}

    for key in params:
        g = grads[key]

        new_m[key] = beta1 * m[key] + (1-beta1) * g
        new_v[key] = beta2 * v[key] + (1-beta2) * g**2

        m_hat = new_m[key] / (1-beta1**t)
        v_hat = new_v[key] / (1-beta2**t)

        new_params[key] = params[key] - lr * m_hat/(np.sqrt(v_hat) + eps)
    
    return new_params, new_m, new_v

def adamw_update(params, grads, m, v, t, lr, beta1, beta2, eps, wd):
    new_params ={}
    new_m ={}
    new_v = {}

    for key in params:
        g = grads[key]
        new_m[key] = beta1*m[key] + (1-beta1)*g
        new_v[key] = beta2*v[key] + (1-beta2)*g**2
        m_hat = new_m[key]/(1-beta1**t)
        v_hat = new_v[key]/(1-beta2**t)

        adam_step = m_hat/(np.sqrt(v_hat)+eps)

        decay_step = wd*params[key]

        new_params[key] = params[key] - lr * (adam_step + decay_step)

    return new_params, new_m, new_v
    
def train(optimser_name, epochs=500, lr=0.01, beta1=0.9, beta2=0.999, eps= 1e-8, wd=0.01):

    params = init_params()

    m = {k: np.zeros_like(v) for k,v in params.items()}

    v = {k: np.zeros_like(v) for k,v in params.items()}

    velocity = {k: np.zeros_like(v) for k,v in params.items()}

    losses =[]

    for epoch in range(1,epochs+1):

        loss, grads = forward_backward(params, X_tr, Y_tr)
        losses.append(loss)

        t = epoch
        
        if optimser_name == 'sgd':
            params = sgd_update(params, grads, lr)

        elif optimser_name =='momentum':
            params, velocity = momentum_update(params, grads, velocity, beta1, lr)

        elif optimser_name == 'adam':
            params, m, v = adam_update(params, grads, m,v,t,lr, beta1, beta2, eps)
        
        elif optimser_name == 'adamw':
            params, m,v = adamw_update(params, grads, m,v,t,lr, beta1, beta2, eps, wd)

        if epoch%100 ==0:
            preds = (forward_backward(params, X_te, Y_te)[0],)
            print(f"[{optimser_name.upper():8s}] "
                  f"Epoch{epoch:4d}  |  Train Loss:{loss:.4f}")
            
    _,_ = forward_backward(params, X_tr, Y_tr)
    Z1 = params['W1'] @ X_te + params['b1']
    A1 = sigmoid(Z1)
    Z2 = params['W2'] @ A1 + params['b2']
    A2 = sigmoid(Z2)
    preds = (A2>=0.5).astype(int)
    acc = np.mean(preds == Y_te) * 100
    print(f"[{optimser_name.upper():8s}] Final Test Accuracy : {acc:.1f}%\n")

    return losses

for name, lr in [('sgd',0.1),('momentum',0.05),('adam',0.01),('adamw',0.01)]:
    print(f"{name.upper()} (lr = {lr})")
    losses = train(name, epochs = 500, lr=lr)
    print(f"Initial loss : {losses[0]:.4f}")
    print(f"Final Loss :{losses[-1]:.4f}")