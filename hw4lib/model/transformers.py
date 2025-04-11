import torch.nn as nn
import torch
import random
from typing import Tuple, Optional, Literal
from .masks import PadMask, CausalMask
from .positional_encoding import PositionalEncoding
from .decoder_layers import SelfAttentionDecoderLayer, CrossAttentionDecoderLayer
from .encoder_layers import SelfAttentionEncoderLayer
from .speech_embedding import SpeechEmbedding
import warnings
from torchinfo import summary
'''
TODO: Implement these Modules.

This file contains two key transformer architectures:

1. DecoderOnlyTransformer: Used for language modeling tasks (like GPT)
   - Contains a stack of SelfAttentionDecoderLayers
   - Uses causal masking to prevent attending to future tokens
   - Includes optional weight tying and layer dropout features

    Key components to implement:
    1. Token Embedding Layer: Convert token IDs to vectors
    2. Positional Encoding: Add position information
    3. Decoder Stack: Process tokens sequentially
    4. Output Projection: Convert final representations to logits

    Architecture follows Pre-LN (Layer Normalization) design where:
    - Layer normalization is applied at the start of each sublayer
    - Residual connections wrap around each sublayer
    - Final layer norm is applied before output projection

    Implementation Notes:
    1. The forward pass should handle:
    - Proper masking (both padding and causal)
    - Collecting attention weights from all layers
    - Optional layer dropout during training
    
    2. The score method should:
    - Handle single token prediction
    - Not apply padding masks
    - Return only the final token's logits

2. EncoderDecoderTransformer: Used for ASR (Automatic Speech Recognition) tasks
   - Contains an encoder stack for processing speech features
   - Contains a decoder stack for generating text tokens
   - Uses both self-attention and cross-attention mechanisms
   - Includes CTC auxiliary loss support and optional weight tying

   Key components to implement:
   1. Speech Embedding: Convert speech features to vectors with time reduction
   2. Positional Encoding: Add position information (optional for both encoder/decoder)
   3. Encoder Stack: Process speech features
   4. Decoder Stack: Generate text tokens
   5. CTC Head: For auxiliary CTC loss computation
   6. Output Projection: Convert final representations to logits

   Architecture follows Pre-LN (Layer Normalization) design where:
   - Layer normalization is applied at the start of each sublayer
   - Residual connections wrap around each sublayer
   - Final layer norm is applied before output projection

   Implementation Notes:
   1. The forward pass should handle:
   - Proper masking (padding for encoder, both padding and causal for decoder)
   - Collecting attention weights from all layers
   - Optional layer dropout during training
   - CTC logits computation

   2. The score method should:
   - Handle single token prediction given encoder output
   - Not apply padding masks to decoder inputs
   - Return only the final token's logits
'''

## -------------------------------------------------------------------------------------------------
## Decoder-Only Transformer
## -------------------------------------------------------------------------------------------------
class DecoderOnlyTransformer(nn.Module):
    '''
    A Pre-LN Decoder-Only Transformer model.
    '''
    def __init__(
            self, 
            num_layers: int, 
            d_model: int, 
            num_heads: int, 
            d_ff: int, 
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.max_len         = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes     = num_classes
        self.num_layers      = num_layers

        self.dec_layers = nn.ModuleList([
            SelfAttentionDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.target_embedding    = nn.Embedding(num_classes, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.final_linear        = nn.Linear(d_model, num_classes)
        self.dropout             = nn.Dropout(dropout)
        self.norm                = nn.LayerNorm(d_model)

        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

    def forward(self, padded_targets: torch.Tensor, target_lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        if self.training and target_lengths is None:
            raise ValueError("target_lengths must be provided during training")

        pad_mask_dec = PadMask(padded_targets, target_lengths) if target_lengths is not None else None
        causal_mask = CausalMask(padded_targets)

        x = self.target_embedding(padded_targets)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        running_att = {}
        for i in range(self.num_layers):
            if self.training and self.layer_drop_rate > 0 and random.random() < self.layer_drop_rate:
                continue

            x, attention = self.dec_layers[i](x, pad_mask_dec, causal_mask)
            running_att[f'layer{i+1}_dec_self'] = attention

        x = self.norm(x)
        seq_out = self.final_linear(x)
        return seq_out, running_att

    def score(self, batch_prompts: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ValueError("score method is not supported during training, use forward method instead")
        seq_out, _ = self.forward(batch_prompts, target_lengths=None)
        return seq_out[:, -1, :]

## -------------------------------------------------------------------------------------------------
## Encoder-Decoder Transformer
## -------------------------------------------------------------------------------------------------
class EncoderDecoderTransformer(nn.Module):
    '''
    A Pre-LN Encoder-Decoder Transformer model for ASR tasks.
    '''
    def __init__(
            self,
            input_dim: int,  
            time_reduction: int, 
            reduction_method: Literal['lstm', 'conv', 'both'], 
            num_encoder_layers: int,
            num_encoder_heads: int,
            d_ff_encoder: int, 
            num_decoder_layers: int,
            num_decoder_heads: int,
            d_ff_decoder: int,
            d_model: int,
            dropout: float, 
            max_len: int, 
            num_classes: int,
            weight_tying: bool = False,
            layer_drop_rate: float = 0.0,
            skip_encoder_pe: bool = False,
            skip_decoder_pe: bool = False,
    ):
        '''
        Initialize the Encoder-Decoder Transformer model.

        Args:
            input_dim: int, dimension of input speech features
            time_reduction: int, stride along time dimension, the amount of reduction to apply to the time dimension
            reduction_method: Literal['lstm', 'conv', 'both'], the source_embedding reduction method
            num_encoder_layers: int, number of encoder layers
            num_encoder_heads: int, number of encoder attention heads
            d_ff_encoder: int, feed-forward dimension for encoder
            num_decoder_layers: int, number of decoder layers
            num_decoder_heads: int, number of decoder attention heads
            d_ff_decoder: int, feed-forward dimension for decoder
            d_model: int, model dimension
            dropout: float, dropout rate
            max_len: int, maximum sequence length this model can handle
            num_classes: int, number of classes
            weight_tying: bool, whether to use weight tying (default: False)
            layer_drop_rate: float, layer drop rate (default: 0.0)
            skip_encoder_pe: bool, whether to skip positional encoding for encoder (default: False)
            skip_decoder_pe: bool, whether to skip positional encoding for decoder (default: False)
        '''
        super().__init__()

        # TODO: Implement _init_

        # Initialize model attributes
        # DO NOT MODIFY THESE ATTRIBUTES
        self.max_len = max_len
        self.layer_drop_rate = layer_drop_rate
        self.num_classes = num_classes
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.skip_encoder_pe = skip_encoder_pe
        self.skip_decoder_pe = skip_decoder_pe

        # TODO: Create encoder layers
        # Use ModuleList to create a list of encoder layers
        self.enc_layers = nn.ModuleList([
            SelfAttentionEncoderLayer(d_model, num_encoder_heads, d_ff_encoder, dropout)
            for _ in range(num_encoder_layers)
        ]) # ModuleList of encoder layers

        # TODO: Create decoder layers
        # Use ModuleList to create a list of decoder layers
        self.dec_layers = nn.ModuleList([
            CrossAttentionDecoderLayer(d_model, num_decoder_heads, d_ff_decoder, dropout)
            for _ in range(num_decoder_layers)
        ]) # ModuleList of decoder layers

        # TODO: Create source and target embeddings and other layers
        # Use SpeechEmbedding class to create the source embedding
        self.source_embedding = SpeechEmbedding(input_dim, d_model, time_reduction, reduction_method) # Speech embedding


        # TODO: Create the target embedding
        # Use nn.Embedding class to create the target embedding
        self.target_embedding    = nn.Embedding(num_classes, d_model) # Target embedding

        # TODO: Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(d_model, max_len) # Positional encoding

        # TODO: Create the final linear layer
        self.final_linear        = nn.Linear(d_model, num_classes)# Final linear layer

        # TODO: Create the dropout layer
        self.dropout             = nn.Dropout(dropout) # Dropout

        # TODO: Create the encoder normalization layer
        self.encoder_norm        = nn.LayerNorm(d_model)# Encoder normalization

        # TODO: Create the decoder normalization layer
        self.decoder_norm        = nn.LayerNorm(d_model) # Decoder normalization

        # TODO: Create the CTC head
        # Use nn.Sequential to create the CTC head
        # CTC head should project the final encoder output from the d_model space to the num_classes space
        # To be compatible with CTCLoss, a log_softmax to the output (See. nn.LogSoftmax)
        self.ctc_head            = nn.Sequential(
            nn.Linear(d_model, num_classes),
            nn.LogSoftmax(dim=-1)
        ) # CTC head


        # Weight tying if enabled (extra form of regularization, read more about it)
        if weight_tying:
            self.target_embedding.weight = self.final_linear.weight

        #raise NotImplementedError # Remove once implemented

    def encode(self, padded_sources: torch.Tensor, source_lengths: torch.Tensor):
        x_enc, x_enc_lengths = self.source_embedding(padded_sources, source_lengths)
        if not self.skip_encoder_pe:
            x_enc = self.positional_encoding(x_enc)
        x_enc = self.dropout(x_enc)
        pad_mask_src = PadMask(x_enc, x_enc_lengths)

        running_att = {}
        for i, layer in enumerate(self.enc_layers):
            if self.training and self.layer_drop_rate > 0 and torch.rand(1).item() < self.layer_drop_rate:
                continue
            x_enc, att = layer(x_enc, key_padding_mask=pad_mask_src)
            running_att[f'layer{i+1}_enc_self'] = att

        x_enc = self.encoder_norm(x_enc)
        ctc_logits = self.ctc_head(x_enc.permute(1, 0, 2))
        ctc_inputs = {'log_probs': ctc_logits, 'lengths': x_enc_lengths}
        return x_enc, pad_mask_src, running_att, ctc_inputs

    def decode(
        self,
        padded_targets: torch.Tensor,
        encoder_output: torch.Tensor,
        target_lengths: Optional[torch.Tensor] = None,
        pad_mask_src: Optional[torch.Tensor] = None,
    ):
        pad_mask_tgt = PadMask(padded_targets, target_lengths) if target_lengths is not None else None
        if pad_mask_tgt is None and self.training:
            warnings.warn("pad_mask_tgt is None, provide target_lengths during training")
        causal_mask = CausalMask(padded_targets)

        x_dec = self.target_embedding(padded_targets)
        if not self.skip_decoder_pe:
            x_dec = self.positional_encoding(x_dec)
        x_dec = self.dropout(x_dec)

        running_att = {}
        for i, layer in enumerate(self.dec_layers):
            if self.training and self.layer_drop_rate > 0 and torch.rand(1).item() < self.layer_drop_rate:
                continue
            x_dec, self_attn, cross_attn = layer(
                x_dec, encoder_output, pad_mask_tgt, pad_mask_src, causal_mask
            )
            running_att[f'layer{i+1}_dec_self'] = self_attn
            running_att[f'layer{i+1}_dec_cross'] = cross_attn

        x_dec = self.decoder_norm(x_dec)
        seq_out = self.final_linear(x_dec)
        return seq_out, running_att


    def forward(
        self,
        padded_sources: torch.Tensor,
        padded_targets: torch.Tensor,
        source_lengths: Optional[torch.Tensor] = None,
        target_lengths: Optional[torch.Tensor] = None
    ):
        if self.training and (source_lengths is None or target_lengths is None):
            raise ValueError("source_lengths and target_lengths are required during training")
        encoder_output, pad_mask_src, enc_running_att, ctc_inputs = self.encode(padded_sources, source_lengths)
        seq_out, dec_running_att = self.decode(padded_targets, encoder_output, target_lengths, pad_mask_src)
        running_att = {**enc_running_att, **dec_running_att}
        return seq_out, running_att, ctc_inputs

    def score(self, batch_prompts: torch.Tensor, encoder_output: torch.Tensor, pad_mask_src: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise ValueError("score method is not supported during training")
        seq_out, _ = self.decode(batch_prompts, encoder_output, None, pad_mask_src)
        return seq_out[:, -1, :]


    @classmethod
    def from_pretrained_decoder(
        cls,
        decoder_checkpoint_path: str,
        config: dict,
    ) -> Tuple['EncoderDecoderTransformer', dict]:
        """
        Helper function to initialize an encoder-decoder transformer with decoder weights initialized from a pretrained decoder-only model.
        
        Args:
            decoder_checkpoint_path: Path to decoder-only transformer checkpoint
            config: Configuration dictionary for the encoder-decoder model
            
        Returns:
            model: Initialized encoder-decoder transformer
            param_info: Dictionary containing lists of named parameters {'transferred': [(name, param)], 'new': [(name, param)]}
        """
        print("\n=== Initializing Encoder-Decoder from Pretrained Decoder ===")
        print(f"Loading checkpoint from: {decoder_checkpoint_path}")
        
        # Create new encoder-decoder model
        print("\nCreating new encoder-decoder model...")
        model = cls(**config)

        # Load decoder checkpoint
        print("Loading pretrained decoder weights...")
        checkpoint = torch.load(decoder_checkpoint_path, map_location='cpu', weights_only=True)
        decoder_state_dict = checkpoint['model_state_dict']
        
        # Track named parameters
        transferred_params = []
        new_params = []
        
        def transfer_module_weights(target_module, prefix):
            module_state_dict = {
                k.replace(prefix, ''): v 
                for k, v in decoder_state_dict.items()
                if k.startswith(prefix)
            }
            param_count = sum(p.numel() for p in target_module.parameters())
            print(f"  - Transferring {prefix} ({param_count:,} parameters)")
            target_module.load_state_dict(module_state_dict)
            # Store the full parameter names with their prefix
            for name, param in target_module.named_parameters():
                transferred_params.append((f"{prefix}{name}", param))

        # Transfer shared components
        print("\nTransferring shared components:")
        transfer_module_weights(model.target_embedding, 'target_embedding.')
        transfer_module_weights(model.final_linear, 'final_linear.')
        transfer_module_weights(model.decoder_norm, 'norm.')
        
        # Transfer decoder layers
        num_layers = min(
            len([k for k in decoder_state_dict.keys() if k.startswith('dec_layers.')]) // 2,
            model.num_decoder_layers
        )
        print(f"\nTransferring decoder layers (found {num_layers} layers):")
        
        for i in range(num_layers):
            print(f"\nLayer {i + 1}/{num_layers}:")
            transfer_module_weights(
                model.dec_layers[i].self_attn,
                f'dec_layers.{i}.self_attn.'
            )
            transfer_module_weights(
                model.dec_layers[i].ffn,
                f'dec_layers.{i}.ffn.'
            )
        
        # Collect new parameters with their names
        print("\nCollecting new parameters...")
        for name, param in model.named_parameters():
            is_new = True
            for transferred_name, transferred_param in transferred_params:
                if param is transferred_param:
                    is_new = False
                    break
            if is_new:
                new_params.append((name, param))
        
        print("\n=== Initialization Complete ===")
        return model, {'transferred': transferred_params, 'new': new_params}

    def log_param_groups(self, param_groups: list) -> None:
        """Log information about parameter groups."""
        print("\nParameter groups:")
        total_params = 0
        total_trainable = 0
        
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            trainable = sum(p.numel() for p in group['params'] if p.requires_grad)
            total_params += num_params
            total_trainable += trainable
            
            print(f"\n{group['name']}:")
            print(f"  Parameters: {num_params:,}")
            print(f"  Trainable: {trainable:,}")
            print(f"  LR factor: {group['lr_factor']}")
        
        print(f"\nTotal parameters: {total_params:,}")
        print(f"Total trainable: {total_trainable:,}")


## -------------------------------------------------------------------------------------------------
## Test Cases
## -------------------------------------------------------------------------------------------------

def get_decoder_only_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def get_encoder_decoder_inputs(max_len: int = 300, num_classes: int = 10000):
    batch_size = 8
    padded_targets = torch.randint(0, num_classes, (batch_size, max_len))
    source_lengths = torch.ones(batch_size) * max_len
    return padded_targets, source_lengths


def test_decoder_only(num_layers: int = 12, num_heads: int = 8, d_model: int = 512, d_ff: int = 2048, dropout: float = 0.1, max_len: int = 300, num_classes: int = 1000):
    padded_targets, target_lengths = get_decoder_only_inputs(max_len, num_classes)
    model = DecoderOnlyTransformer(num_layers, d_model, num_heads, d_ff, dropout, max_len, num_classes)
    summary(model, input_data=[padded_targets, target_lengths])

if __name__ == "_main_":
    test_decoder_only()