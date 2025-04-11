import torch

''' 
TODO: Implement this function.

Specification:
- Function should create a padding mask that identifies padded positions in the input
- Mask should be a boolean tensor of shape (N, T) where:
  * N = batch size from padded_input
  * T = sequence length from padded_input
- True values indicate padding positions that should be masked
- False values indicate valid positions that should not be masked
- Padding is assumed to be on the right side of sequences
- Each sequence in the batch may have different valid lengths
- Mask should be on same device as input tensor
'''
def PadMask(padded_input, input_lengths):
    """ 
    Create a mask to identify non-padding positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
        input_lengths: The actual lengths of each sequence before padding, shape (N,).
    Returns:
        A boolean mask tensor with shape (N, T), where: 
            - padding positions are marked with True 
            - non-padding positions are marked with False.
    """
    # TODO: Implement PadMask
    batch_size = padded_input.size(0)
    seq_length = padded_input.size(1)
    
    # Create a range tensor for the sequence positions
    # Shape: (1, seq_length)
    positions = torch.arange(seq_length, device=padded_input.device).unsqueeze(0)
    
    # Expand the positions tensor to match the batch size
    # Shape: (batch_size, seq_length)
    positions = positions.expand(batch_size, -1)
    
    # Expand the input_lengths tensor to have the right shape for comparison
    # Shape: (batch_size, 1)
    lengths = input_lengths.unsqueeze(1)
    
    # Create the mask by comparing positions with lengths
    # Positions >= lengths are padding (True), otherwise they are valid (False)
    mask = positions >= lengths
    
    return mask

''' 
TODO: Implement this function.

Specification:
- Function should create a causal mask for self-attention
- Mask should be a boolean tensor of shape (T, T) where T is sequence length
- True values indicate positions that should not attend to each other
- False values indicate positions that can attend to each other
- Causal means each position can only attend to itself and previous positions
- Mask should be on same device as input tensor
- Mask should be upper triangular (excluding diagonal)
'''
def CausalMask(padded_input):
    """ 
    Create a mask to identify non-causal positions. 
    Args:
        padded_input: The input tensor with padding, shape (N, T, ...) or (N, T).
    
    Returns:
        A boolean mask tensor with shape (T, T), where: 
            - non-causal positions (don't attend to) are marked with True 
            - causal positions (can attend to) are marked with False.
    """
    # TODO: Implement CausalMask
    if padded_input.dim() > 2:
        # Input is (batch_size, seq_length, feature_dim)
        seq_length = padded_input.size(1)
    else:
        # Input is (batch_size, seq_length)
        seq_length = padded_input.size(1)
    
    # Create indices for rows and columns
    i = torch.arange(seq_length, device=padded_input.device)
    j = torch.arange(seq_length, device=padded_input.device)
    
    # Create a mask where each position can only attend to itself and previous positions
    # This means we want to mask out (set to True) positions where j > i
    # Create grid indices
    i, j = torch.meshgrid(i, j, indexing='ij')
    
    # Create the mask: True for positions that should NOT attend to each other
    # (upper triangular part excluding diagonal)
    mask = j > i
    
    return mask

