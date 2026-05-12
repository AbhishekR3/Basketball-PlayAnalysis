'''
nn/model.py

Neural network components: MultiHeadAttention with prunable heads, and the
BasketballLSTM classifier (bidirectional LSTM + self-attention + sigmoid).
'''


#%% Import libraries

import torch
import torch.nn as nn

from ._logger import logger


#%% Multi-Head Self-Attention

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism for sequence modeling with prunable heads"""

    def __init__(self, hidden_dim, num_heads=8):
        """
        Objective:
        Initialize the multi-head self-attention module with prunable heads

        Parameters:
        [int] hidden_dim - Dimension of the hidden state
        [int] num_heads - Number of attention heads (default: 8)
        """
        try:
            super(MultiHeadAttention, self).__init__()
            self.hidden_dim = hidden_dim
            self.num_heads = num_heads
            self.head_dim = hidden_dim // num_heads

            # Ensure hidden_dim is divisible by num_heads
            assert hidden_dim % num_heads == 0, "Hidden dimension must be divisible by number of heads"

            # Define the attention components
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)

            # Output projection
            self.output_projection = nn.Linear(hidden_dim, hidden_dim)

            # Scale factor for attention scores
            self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

            # Head importance scores (for pruning)
            self.head_importance = nn.Parameter(torch.ones(num_heads, 1))

        except Exception as e:
            logger.error(f"Error initializing MultiHeadAttention: {e}")
            raise

    def forward(self, hidden_state):
        """
        Objective:
        Forward pass of the multi-head self-attention module

        Parameters:
        [torch.Tensor] hidden_state - Input hidden state (batch_size, seq_len, hidden_dim)

        Returns:
        [torch.Tensor] attended - Attention-weighted output
        [torch.Tensor] attention_weights - Attention weights for all heads
        """
        try:
            batch_size = hidden_state.shape[0]
            seq_len = hidden_state.shape[1]

            # Move scale to the same device as hidden state
            self.scale = self.scale.to(hidden_state.device)

            # Linear projections
            Q = self.query(hidden_state)  # (batch_size, seq_len, hidden_dim)
            K = self.key(hidden_state)    # (batch_size, seq_len, hidden_dim)
            V = self.value(hidden_state)  # (batch_size, seq_len, hidden_dim)

            # Reshape for multi-head attention
            Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            # Calculate attention scores
            energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

            # Apply softmax to get attention weights
            attention = torch.softmax(energy, dim=-1)

            # Store attention weights for later use
            attention_weights = attention

            # Apply head importance (for pruning) - MODIFIED FOR 2D
            # Reshape head_importance for broadcasting [num_heads, 1] -> [1, num_heads, 1, 1]
            head_importance = self.head_importance.view(1, self.num_heads, 1, 1)
            attention = attention * head_importance

            # Apply attention weights to values
            weighted_V = torch.matmul(attention, V)

            # Reshape back to original dimensions
            weighted_V = weighted_V.permute(0, 2, 1, 3).contiguous()
            weighted_V = weighted_V.view(batch_size, seq_len, self.hidden_dim)

            # Apply output projection
            attended = self.output_projection(weighted_V)

            return attended, attention_weights

        except Exception as e:
            logger.error(f"Error in MultiHeadAttention.forward: {e}")
            raise


#%% Bidirectional LSTM Classifier

class BasketballLSTM(nn.Module):
    """LSTM model for basketball play classification with prunable attention"""

    def __init__(self, input_dim, hidden_dim=192, output_dim=1, num_layers=2,
                 dropout=0.3, recurrent_dropout=0.15, bidirectional=True, num_heads=8):
        """
        Objective:
        Initialize the LSTM model for basketball play classification with prunable attention

        Parameters:
        [int] input_dim - Dimension of input features
        [int] hidden_dim - Dimension of LSTM hidden state (default: 192)
        [int] output_dim - Dimension of output (1 for binary classification)
        [int] num_layers - Number of LSTM layers (default: 2)
        [float] dropout - Dropout probability (default: 0.3)
        [float] recurrent_dropout - Recurrent dropout probability (default: 0.15)
        [bool] bidirectional - Whether to use bidirectional LSTM (default: True)
        [int] num_heads - Number of attention heads (default: 8)
        """
        try:
            super(BasketballLSTM, self).__init__()

            self.hidden_dim = hidden_dim
            self.num_layers = num_layers
            self.bidirectional = bidirectional
            self.num_directions = 2 if bidirectional else 1

            # LSTM layers
            self.lstm = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )

            # Apply recurrent dropout
            self.recurrent_dropout = nn.Dropout(recurrent_dropout)

            # Multi-head self-attention layer (prunable)
            attention_dim = hidden_dim * self.num_directions
            self.attention = MultiHeadAttention(attention_dim, num_heads=num_heads)

            # Layer normalization
            self.layer_norm = nn.LayerNorm(attention_dim)

            # Output layer
            self.fc = nn.Linear(attention_dim, output_dim)

            # Sigmoid activation for binary classification
            self.sigmoid = nn.Sigmoid()

            logger.info(f"Initialized BasketballLSTM model with {input_dim} input features, "
                        f"{hidden_dim} hidden dim, {num_layers} layers, "
                        f"bidirectional={bidirectional}, attention_heads={num_heads}")

        except Exception as e:
            logger.error(f"Error initializing BasketballLSTM: {e}")
            raise

    def forward(self, x, hidden=None):
        """
        Objective:
        Forward pass of the LSTM model

        Parameters:
        [torch.Tensor] x - Input sequence of shape (batch_size, seq_len, input_dim)
        [tuple] hidden - Initial hidden state (optional)

        Returns:
        [torch.Tensor] out - Output prediction
        [tuple] hidden - Final hidden state
        """
        try:
            batch_size = x.size(0)

            # Initialize hidden state if not provided
            if hidden is None:
                h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
                hidden = (h0, c0)

            # LSTM forward
            lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)

            # Apply recurrent dropout to the output
            lstm_out = self.recurrent_dropout(lstm_out)

            # Apply multi-head self-attention
            attended, _ = self.attention(lstm_out)

            # Residual connection and layer normalization
            attended = self.layer_norm(lstm_out + attended)

            # Use the average of the attended output across the sequence
            out = attended.mean(dim=1)

            # Final prediction
            out = self.fc(out)
            out = self.sigmoid(out)

            return out, hidden

        except Exception as e:
            logger.error(f"Error in BasketballLSTM.forward: {e}")
            raise
