import numpy as np
np.random.seed(42)


# Dropout

class Dropout:

    def __init__(self, p=0.5):
        assert 0.0<=p<=1.0 # p must be between 0 and 1
        self.p = p
        self.training = True
        self.mask = None

    def forward(self, x):
        if not self.training or self.p ==0:
            # if inference mode or p=0, must pass unchanged
            return x
        
        #Now sampling a binary mask, each element is independently chosen : Bernoulli (1-p)
        self.mask = (np.random.rand(*x.shape)>self.p).astype(float)

        # Applying mask and also scaling to keep expected value of output equal to input
        return self.mask * x / (1.0-self.p)
    
    def backward(self, dout):

        if not self.training or self.p == 0.0:
            # gradient passes unchanged at inference
            return dout
        
        # gradient is zeroed for dropped neurons and scaled for survivors
        return self.mask * dout/(1.0 - self.p)
    
    def eval(self):
        #switching to inference mode
        self.training = False

    def train(self):
        #switching to training mode
        self.training = True



# Batch Normalization

class BatchNorm1d:

    def __init__ (self, d, eps=1e-5, momentum=0.1):
        self.d = d # number of features (neurons in this layer)
        self.eps = eps
        self.momentum = momentum # needed for running stats update
        
        #initialising to gamma=1 and beta=0 for no effect
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        
        #Running statistics
        self.running_mean = np.zeros(d) #accumulated mean
        self.running_var = np.ones(d) # accumulated variance

        self.training = True #mode flag
        self.cache = None #saved values of backward pass
    
    def forward(self,x):

        if self.training:
            mu = x.mean(axis = 0) # mean across samples
            var = x.var(axis=0)  # variance across samples

            x_hat = (x-mu)/np.sqrt(var+self.eps) # normalization

            # calculating running mean and var to accumulate them for evaluation time
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mu
            self.running_var = (1-self.momentum)*self.running_var + self.momentum*var

            self.cache = (x,x_hat,mu,var)
        
        else :
             x_hat = (x-self.running_mean) / np.sqrt (self.running_var + self.eps)

             # Affine transform : scale and shift with learnable params
             return self.gamma * x_hat + self.beta
        
    def backward (self, dout):

        x, x_hat, mu, var = self.cache
        m = x.shape[0]

        self.dgamma = np.sum(dout*x_hat, axis=0)
        self.dbeta = np.sum(dout,axis=0)

        dx_hat = dout * self.gamma

        std_inv = 1.0/np.sqrt(var+self.eps)
        dvar = np.sum(dx_hat * (x-mu) * -0.5 * std_inv**3, axis=0)

        dmu = (np.sum(dx_hat * (-std_inv), axis=0) + dvar*np.mean(-2.0 *(x-mu),axis=0))

        dx = (dx_hat * std_inv + dvar * 2.0 * (x-mu) /m + dmu/m)

        return dx

    def eval(self):
        self.training = False
    
    def train(self):
        self.training = True

    




