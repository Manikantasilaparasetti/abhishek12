"""
Recurrent Neural Network components (RNN, LSTM, GRU).
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from ..tensor import Tensor
from .modules import Module
from .activations import tanh, relu, sigmoid


class RNNCell(Module):
    """Vanilla RNN cell with configurable activation."""
    
    def __init__(self, input_size: int, hidden_size: int, activation: str = "tanh"):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + hidden_size))
        
        # Input-to-hidden weights
        self.weight_ih = Tensor(
            np.random.uniform(-limit, limit, (input_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Hidden-to-hidden weights
        self.weight_hh = Tensor(
            np.random.uniform(-limit, limit, (hidden_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Bias
        self.bias = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)
        
        self._parameters.extend([self.weight_ih, self.weight_hh, self.bias])
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """Forward pass through RNN cell."""
        # Input validation
        if x.data.ndim != 2:
            raise ValueError(f"RNNCell expects 2D input (batch_size, input_size), got {x.data.ndim}D with shape {x.shape}")
        
        batch_size, input_features = x.shape
        
        if input_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {input_features}")
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        # Validate hidden state shape
        if hidden.shape != (batch_size, self.hidden_size):
            raise ValueError(f"Hidden state shape mismatch: expected {(batch_size, self.hidden_size)}, got {hidden.shape}")
        
        # Compute: new_hidden = activation(x @ W_ih + hidden @ W_hh + bias)
        input_part = x.matmul(self.weight_ih)
        hidden_part = hidden.matmul(self.weight_hh)
        
        preactivation = input_part + hidden_part + self.bias
        
        # Apply activation function
        if self.activation == "tanh":
            new_hidden = tanh(preactivation)
        elif self.activation == "relu":
            new_hidden = relu(preactivation)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        
        return new_hidden
    
    def __repr__(self):
        return (f"RNNCell(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"activation={self.activation})")


class LSTMCell(Module):
    """LSTM cell with forget, input, and output gates."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + hidden_size))
        
        # Input-to-hidden weights for all gates (input_size, 4*hidden_size)
        # Order: [input_gate, forget_gate, candidate_gate, output_gate]
        self.weight_ih = Tensor(
            np.random.uniform(-limit, limit, (input_size, 4*hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Hidden-to-hidden weights for all gates (hidden_size, 4*hidden_size)
        self.weight_hh = Tensor(
            np.random.uniform(-limit, limit, (hidden_size, 4*hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Biases for all gates (4*hidden_size,)
        self.bias_ih = Tensor(np.zeros(4*hidden_size, dtype=np.float32), requires_grad=True)
        self.bias_hh = Tensor(np.zeros(4*hidden_size, dtype=np.float32), requires_grad=True)
        
        self._parameters.extend([self.weight_ih, self.weight_hh, self.bias_ih, self.bias_hh])
    
    def forward(self, x: Tensor, states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass through LSTM cell."""
        # Input validation
        if x.data.ndim != 2:
            raise ValueError(f"LSTMCell expects 2D input (batch_size, input_size), got {x.data.ndim}D with shape {x.shape}")
        
        batch_size, input_features = x.shape
        
        if input_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {input_features}")
        
        # Initialize states if not provided
        if states is None:
            h_prev = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
            c_prev = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        else:
            h_prev, c_prev = states
        
        # Validate state shapes
        if h_prev.shape != (batch_size, self.hidden_size):
            raise ValueError(f"Hidden state shape mismatch: expected {(batch_size, self.hidden_size)}, got {h_prev.shape}")
        if c_prev.shape != (batch_size, self.hidden_size):
            raise ValueError(f"Cell state shape mismatch: expected {(batch_size, self.hidden_size)}, got {c_prev.shape}")
        
        # Compute all gates at once
        # x @ W_ih + h_prev @ W_hh + bias
        gi = x.matmul(self.weight_ih) + self.bias_ih
        gh = h_prev.matmul(self.weight_hh) + self.bias_hh
        gates = gi + gh
        
        # Split gates: [input, forget, candidate, output]
        i_gate, f_gate, c_gate, o_gate = self._split_gates_fixed(gates)
        
        # Apply activations
        i_gate = sigmoid(i_gate)  # Input gate
        f_gate = sigmoid(f_gate)  # Forget gate
        c_gate = tanh(c_gate)     # Candidate values
        o_gate = sigmoid(o_gate)  # Output gate
        
        # Update cell state: c_new = f_gate * c_prev + i_gate * c_gate
        c_new = f_gate * c_prev + i_gate * c_gate
        
        # Update hidden state: h_new = o_gate * tanh(c_new)
        h_new = o_gate * tanh(c_new)
        
        return h_new, c_new
    
    def _split_gates_fixed(self, gates: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Split concatenated gates into individual gate tensors."""
        # gates shape: (batch_size, 4*hidden_size)
        batch_size = gates.shape[0]
        
        # Split into 4 equal parts
        gate_data = gates.data.reshape(batch_size, 4, self.hidden_size)
        
        i_gate = Tensor(gate_data[:, 0, :], requires_grad=gates.requires_grad)
        f_gate = Tensor(gate_data[:, 1, :], requires_grad=gates.requires_grad)
        c_gate = Tensor(gate_data[:, 2, :], requires_grad=gates.requires_grad)
        o_gate = Tensor(gate_data[:, 3, :], requires_grad=gates.requires_grad)
        
        # Set up backward pass for gate splitting
        def setup_gate_backward(gate_tensor, gate_idx):
            if gates.requires_grad:
                gate_tensor.op = f"GateSplitBackward({gate_idx})"
                gate_tensor.is_leaf = False
                
                def _backward():
                    if gates.grad is None:
                        gates.grad = np.zeros_like(gates.data)
                    gates.grad[:, gate_idx*self.hidden_size:(gate_idx+1)*self.hidden_size] += gate_tensor.grad
                
                gate_tensor._backward = _backward
                gate_tensor.prev = {gates}
        
        setup_gate_backward(i_gate, 0)
        setup_gate_backward(f_gate, 1)
        setup_gate_backward(c_gate, 2)
        setup_gate_backward(o_gate, 3)
        
        return i_gate, f_gate, c_gate, o_gate
    
    def __repr__(self):
        return f"LSTMCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class GRUCell(Module):
    """GRU cell with reset and update gates."""
    
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        limit = np.sqrt(6.0 / (input_size + hidden_size))
        
        # Weights for reset and update gates
        self.weight_ih_rz = Tensor(
            np.random.uniform(-limit, limit, (input_size, 2*hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        self.weight_hh_rz = Tensor(
            np.random.uniform(-limit, limit, (hidden_size, 2*hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Weights for new gate
        self.weight_ih_n = Tensor(
            np.random.uniform(-limit, limit, (input_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        self.weight_hh_n = Tensor(
            np.random.uniform(-limit, limit, (hidden_size, hidden_size)).astype(np.float32),
            requires_grad=True
        )
        
        # Biases
        self.bias_ih_rz = Tensor(np.zeros(2*hidden_size, dtype=np.float32), requires_grad=True)
        self.bias_hh_rz = Tensor(np.zeros(2*hidden_size, dtype=np.float32), requires_grad=True)
        self.bias_ih_n = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)
        self.bias_hh_n = Tensor(np.zeros(hidden_size, dtype=np.float32), requires_grad=True)
        
        self._parameters.extend([
            self.weight_ih_rz, self.weight_hh_rz, self.weight_ih_n, self.weight_hh_n,
            self.bias_ih_rz, self.bias_hh_rz, self.bias_ih_n, self.bias_hh_n
        ])
    
    def forward(self, x: Tensor, hidden: Optional[Tensor] = None) -> Tensor:
        """Forward pass through GRU cell."""
        # Input validation
        if x.data.ndim != 2:
            raise ValueError(f"GRUCell expects 2D input (batch_size, input_size), got {x.data.ndim}D with shape {x.shape}")
        
        batch_size, input_features = x.shape
        
        if input_features != self.input_size:
            raise ValueError(f"Expected {self.input_size} input features, got {input_features}")
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
        
        # Validate hidden state shape
        if hidden.shape != (batch_size, self.hidden_size):
            raise ValueError(f"Hidden state shape mismatch: expected {(batch_size, self.hidden_size)}, got {hidden.shape}")
        
        # Compute reset and update gates
        gi_rz = x.matmul(self.weight_ih_rz) + self.bias_ih_rz
        gh_rz = hidden.matmul(self.weight_hh_rz) + self.bias_hh_rz
        gates_rz = gi_rz + gh_rz
        
        # Split into reset and update gates
        reset_gate_data = gates_rz.data[:, :self.hidden_size]
        update_gate_data = gates_rz.data[:, self.hidden_size:]
        
        reset_gate = sigmoid(Tensor(reset_gate_data, requires_grad=gates_rz.requires_grad))
        update_gate = sigmoid(Tensor(update_gate_data, requires_grad=gates_rz.requires_grad))
        
        # Compute new gate
        gi_n = x.matmul(self.weight_ih_n) + self.bias_ih_n
        gh_n = (reset_gate * hidden).matmul(self.weight_hh_n) + self.bias_hh_n
        new_gate = tanh(gi_n + gh_n)
        
        # Update hidden state: h_new = (1 - update_gate) * new_gate + update_gate * hidden
        one_tensor = Tensor(np.ones_like(update_gate.data))
        one_minus_update = one_tensor - update_gate
        new_hidden = one_minus_update * new_gate + update_gate * hidden
        
        return new_hidden
    
    def __repr__(self):
        return f"GRUCell(input_size={self.input_size}, hidden_size={self.hidden_size})"


class RNN(Module):
    """Multi-layer RNN with support for different cell types."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 cell_type: str = "rnn", dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell_type = cell_type.lower()
        self.dropout = dropout
        self.batch_first = batch_first
        
        # Validate inputs
        if self.cell_type not in ["rnn", "lstm", "gru"]:
            raise ValueError(f"Unsupported cell type '{self.cell_type}'. Choose from 'rnn', 'lstm', 'gru'")
        
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be between 0 and 1, got {dropout}")
        
        # Create RNN layers
        self.layers = []
        for i in range(num_layers):
            # First layer uses input_size, subsequent layers use hidden_size
            layer_input_size = input_size if i == 0 else hidden_size
            
            if self.cell_type == "rnn":
                layer = RNNCell(layer_input_size, hidden_size)
            elif self.cell_type == "lstm":
                layer = LSTMCell(layer_input_size, hidden_size)
            elif self.cell_type == "gru":
                layer = GRUCell(layer_input_size, hidden_size)
            
            self.layers.append(layer)
            self._modules.append(layer)
        
        # Dropout layer (applied between RNN layers, not time steps)
        if dropout > 0:
            from .modules import Dropout
            self.dropout_layer = Dropout(dropout)
            self._modules.append(self.dropout_layer)
        else:
            self.dropout_layer = None
    
    def forward(self, x: Tensor, initial_states: Optional[List] = None) -> Tuple[Tensor, List]:
        """Forward pass through multi-layer RNN."""
        # Input validation
        if x.data.ndim != 3:
            raise ValueError(f"RNN expects 3D input (batch, seq_len, features) or (seq_len, batch, features), got {x.data.ndim}D with shape {x.shape}")
        
        # Handle batch_first format
        if not self.batch_first:
            # Convert from (seq_len, batch, features) to (batch, seq_len, features)
            x = x.transpose((1, 0, 2))
        
        batch_size, seq_len, input_features = x.shape
        
        # Initialize states if not provided
        if initial_states is None:
            current_states = self._init_states(batch_size)
        else:
            current_states = initial_states
        
        # Process each time step
        outputs = []
        for t in range(seq_len):
            # Get input at time step t: (batch_size, input_features)
            x_t = Tensor(x.data[:, t, :], requires_grad=x.requires_grad)
            
            layer_input = x_t
            new_states = []
            
            # Pass through each layer
            for layer_idx, layer in enumerate(self.layers):
                current_state = current_states[layer_idx]
                
                if self.cell_type == "lstm":
                    h_new, c_new = layer(layer_input, current_state)
                    new_states.append((h_new, c_new))
                    layer_input = h_new  # Use hidden state as input to next layer
                else:  # rnn or gru
                    h_new = layer(layer_input, current_state)
                    new_states.append(h_new)
                    layer_input = h_new  # Use hidden state as input to next layer
                
                # Apply dropout between layers (not on last layer)
                if (self.dropout_layer is not None and 
                    layer_idx < len(self.layers) - 1 and 
                    self.training):
                    layer_input = self.dropout_layer(layer_input)
            
            # Store output from last layer at this time step
            outputs.append(layer_input)
            
            # Update states for next time step
            current_states = new_states
        
        # Stack outputs into proper tensor shape
        output_data = np.stack([out.data for out in outputs], axis=1)
        output_tensor = Tensor(output_data, requires_grad=any(out.requires_grad for out in outputs))
        
        # Handle batch_first format for output
        if not self.batch_first:
            # Convert back to (seq_len, batch, features)
            output_tensor = output_tensor.transpose((1, 0, 2))
        
        return output_tensor, current_states
    
    def _init_states(self, batch_size: int) -> List:
        """Initialize hidden states for all layers."""
        states = []
        for _ in range(self.num_layers):
            if self.cell_type == "lstm":
                # LSTM needs both hidden and cell states
                h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
                c = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
                states.append((h, c))
            else:
                # RNN and GRU only need hidden state
                h = Tensor(np.zeros((batch_size, self.hidden_size), dtype=np.float32))
                states.append(h)
        return states
    
    def __repr__(self):
        return (f"RNN(input_size={self.input_size}, hidden_size={self.hidden_size}, "
                f"num_layers={self.num_layers}, cell_type={self.cell_type}, "
                f"dropout={self.dropout}, batch_first={self.batch_first})")
