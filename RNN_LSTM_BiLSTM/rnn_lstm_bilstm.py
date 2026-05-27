import numpy as np
np.random.seed(42)

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def tanh_fn(z):
    return np.tanh(z)

# Vanilla RNN Cell

class VanillaRNN:

    def __init__(self, input_dim, hidden_dim, output_dim):
        self.H = hidden_dim # saving for use in forward pass

        self.W_hh = np.random.randn(hidden_dim, hidden_dim) * 0.01 # maps previous hidden state to hidden space
        self.W_xh = np.random.randn(hidden_dim, input_dim) * 0.01 # maps input to hidden space
        self.W_bh = np.zeros(hidden_dim) # hidden layer bias
        self.W_hy = np.random.randn(output_dim, hidden_dim) * 0.01 # maps hidden state to output space
        self.b_y = np.zeros(output_dim) #output bias

    def forward(self, X):
        # X is a input sequence, shape (T, D), T = seq length, D = input dim
        # return list of hidden states and list of outputs

        T = X.shape[0]
        h = np.zeros(self.H)

        hidden_states = []
        outputs = []

        for t in range(T):
            x_t = X[t] # current input vector, shape (D,)

            # Here h[t] = tanh(Whh*h[t-1] + Wxh * x + bh)
            recurrent = self.W_hh @ h
            input_contrib = self.W_xh @ x_t
            h = tanh_fn (recurrent + input_contrib + self.b_h)

            # Here y[t] = Why*h[t] + by
            y_t = self.W_hy @ h + self.b_y

            # We are saving a copy, not a reference
            hidden_states.append(h.copy())
            outputs.append(y_t.copy())

        return np.array(hidden_states), np.array(outputs)


#LSTM Cell

class LSTMCell:
    # A single LSTM cell that process one time step. 
    
    def __init__(self, input_dim, hidden_dim):
        self.H = hidden_dim
        self.D = input_dim
        concat_dim = hidden_dim + input_dim

        # Forget gate parameters
        self.W_f = np.random.randn(hidden_dim, concat_dim) *0.1
        self.b_f = np.zeros(hidden_dim)
        # Intializing forget bias to 1, common trick to start by remembering
        self.b_f[:]=1

        # Input Gate parameters
        self.W_i = np.random.randn(hidden_dim,concat_dim)*0.1
        self.b_i = np.zeros(hidden_dim)

        # Cell candidate parameters
        self.W_c = np.random.randn(hidden_dim, concat_dim) *0.1
        self.b_c = np.zeros(hidden_dim)

        # Output Gate Parameters
        self.W_o = np.random.randn(hidden_dim, concat_dim) * 0.1
        self.b_o = np.zeros(hidden_dim)

    def step(self, x_t, h_prev, c_prev):
        
        #Concatenating previous hidden state and current input
        s = np.concatenate([h_prev,x_t])

        # Forget gate
        f_t  = sigmoid(self.W_f @ s + self.b_f)

        # Input gate
        i_t = sigmoid(self.W_i @ s + self.b_i)

        # Cell Candidate
        c_tilde = tanh_fn(self.W_c @ s + self.b_c)
        
        # Output Gate
        o_t = sigmoid(self.W_o @ s + self.b_o)

        # Cell state update ( the gradient highway)
        c_t = f_t*c_prev + i_t * c_tilde

        # hidden state
        h_t = o_t * tanh_fn(c_t)

        # the gates values are also returned, which may be useful in debugging
        gates = {'f':f_t, 'i':i_t, 'c_tilde':c_tilde, 'o':o_t}
        return h_t, c_t, gates

class LSTM :

    def __int__(self, input_dim, hidden_dim):
        self.H = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim)

    def forward(self, X):
        # processing full sequence X of shape (T,D)
        T = X.shape[0]

        #intialize both states to zero vectors at t=0
        h = np.zeros(self.H)
        c = np.zeros(self.H)

        all_h = []
        all_c = []

        all_gates = []

        for t in range(T):
            h,c,gates = self.cell.step(X[t], h, c)
            all_h.append(h.copy())
            all_c.append(c.copy())
            all_gates.append(gates)

        return np.array(all_h), np.array(all_c), all_gates

# Bidirectional LSTM

class BiLSTM :
    
    def __int__(self, input_dim, hidden_dim):
        self.H = hidden_dim
        self.fwd_lstm =  LSTM(input_dim, hidden_dim)
        self.bwd_lstm = LSTM(input_dim, hidden_dim)

    def forward (self, X):
        fwd_h, fwd_c, _ = self.fwd_lstm.forward(X)
        bwd_h_flipped,_, _ = self.bwd_lstm.forward(np.flip(X,axis=0))

        bwd_h = np.flip(bwd_h_flipped,axis=0)

        bi_h = np.concatenate([fwd_h,bwd_h],axis=1)

        return bi_h, fwd_h, bwd_h
