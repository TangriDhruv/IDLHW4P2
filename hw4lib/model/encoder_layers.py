import torch.nn as nn
import torch
from typing import Tuple, Optional
from .sublayers import SelfAttentionLayer, FeedForwardLayer

class SelfAttentionEncoderLayer(nn.Module):
    '''
    Pre-LN Encoder Layer with self-attention mechanism.
    Used in the encoder part of transformer architectures.
    '''
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        '''
        Initialize the SelfAttentionEncoderLayer.
        Args:
            d_model (int): The dimension of the model.
            num_heads (int): The number of attention heads.
            d_ff (int): The dimension of the feedforward network.
            dropout (float): The dropout rate.
        '''
        super().__init__()
        # TODO: Implement init
        # TODO: Initialize the sublayers
        
        # Layer normalization layers for Pre-LN architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Self-attention layer
        self.self_attn = SelfAttentionLayer(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForwardLayer(d_model, d_ff, dropout)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass for the EncoderLayer.
        Args:
            x (torch.Tensor): The input tensor. shape: (batch_size, seq_len, d_model)
            key_padding_mask (torch.Tensor): The padding mask for the input. shape: (batch_size, seq_len)
        Returns:
            x (torch.Tensor): The output tensor. shape: (batch_size, seq_len, d_model)
            mha_attn_weights (torch.Tensor): The attention weights. shape: (batch_size, seq_len, seq_len)
        '''
        # TODO: Implement forward: Follow the figure in the writeup
        # What will be different from decoder self-attention layer?
        
        # First sublayer: Self-attention with Pre-LN
        residual = x
        norm1 = self.norm1(x)
        attn_output, mha_attn_weights = self.self_attn(norm1, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(attn_output)
        
        # Second sublayer: Feed-forward with Pre-LN
        residual = x
        norm2 = self.norm2(x)
        ffn_output = self.ffn(norm2)
        x = ffn_output  # Direct FFN output without residual connection
        
        return x, mha_attn_weights