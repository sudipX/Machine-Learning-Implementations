# Recurrent Neural Networks from Scratch using NumPy

A beginner-friendly educational implementation of:

- Vanilla RNN
- LSTM
- Bidirectional LSTM (BiLSTM)

using only NumPy.

This project focuses on understanding the internal mathematics and architecture of recurrent neural networks without using deep learning frameworks like PyTorch or TensorFlow.

---

# Features

- Pure NumPy implementation
- Vanilla RNN implementation
- LSTM Cell implementation
- Full sequence LSTM
- Bidirectional LSTM
- Step-by-step hidden state computation
- Gate-level LSTM operations
- Educational comments throughout the code
- Sequence processing from scratch

---

# Implemented Architectures

# 1. Vanilla RNN

A basic recurrent neural network where the hidden state is updated at every time step.

### Hidden State Update

```math
h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)
```

### Output Equation

```math
y_t = W_{hy}h_t + b_y
```

---

# 2. Long Short-Term Memory (LSTM)

LSTM solves the vanishing gradient problem of traditional RNNs using:

- Forget Gate
- Input Gate
- Cell State
- Output Gate

---

## Forget Gate

Controls what information should be discarded from the previous cell state.

```math
f_t = \sigma(W_f[h_{t-1}, x_t] + b_f)
```

---

## Input Gate

Controls what new information should be stored.

```math
i_t = \sigma(W_i[h_{t-1}, x_t] + b_i)
```

---

## Candidate Cell State

Creates candidate memory values.

```math
\tilde{C}_t = \tanh(W_c[h_{t-1}, x_t] + b_c)
```

---

## Cell State Update

Main memory update equation.

```math
C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t
```

---

## Output Gate

Determines what part of memory becomes hidden state.

```math
o_t = \sigma(W_o[h_{t-1}, x_t] + b_o)
```

---

## Hidden State

Final hidden representation.

```math
h_t = o_t \odot \tanh(C_t)
```

---

# 3. Bidirectional LSTM (BiLSTM)

Processes the sequence in:

- Forward direction
- Backward direction

Then concatenates both hidden representations.

### Bidirectional Representation

```math
h_t^{bi} = [\overrightarrow{h_t}; \overleftarrow{h_t}]
```

This allows the model to capture:

- Past context
- Future context

simultaneously.

---

# Project Structure

```bash
.
├── rnn_lstm_bilstm.py
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

```bash
python rnn_lstm_bilstm.py
```

---

# Code Overview

## Activation Functions

Implemented manually:

### Sigmoid

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

### Tanh

```math
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

---

# Vanilla RNN Flow

At each time step:

1. Current input is processed
2. Previous hidden state is reused
3. New hidden state is computed
4. Output is generated

The hidden state acts as memory across the sequence.

---

# LSTM Flow

At each time step:

1. Forget irrelevant information
2. Decide what new information to store
3. Update long-term memory
4. Produce hidden state

The cell state acts as a gradient highway that helps preserve long-range dependencies.

---

# Bidirectional LSTM Flow

The sequence is processed:

- Left → Right
- Right → Left

Then both hidden states are concatenated.

Useful for:

- NLP
- Speech Recognition
- Machine Translation
- Sequence Labeling
- Named Entity Recognition

---


