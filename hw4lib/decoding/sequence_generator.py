import torch
import torch.nn as nn
from typing import Tuple, Optional, List, Callable
from ..data import H4Tokenizer

'''
TODO: Implement the `generate_greedy` and optionally the `generate_beam` methods of the `SequenceGenerator` class.

This file implements text generation strategies for transformer language models:

1. Greedy Search: Always selects the most likely next token
   - Simple but can lead to repetitive or suboptimal outputs
   - Useful for deterministic generation

2. Beam Search: Maintains top-k most likely sequences at each step
   - Explores multiple possible sequences in parallel
   - Often produces higher quality outputs than greedy search
   - More computationally intensive

3. Sampling with Filtering: Uses probabilistic sampling with constraints
   - Temperature: Controls randomness of sampling
   - Top-k: Limits sampling to k most likely tokens
   - Top-p (nucleus): Samples from minimal set of tokens comprising p probability mass
   - Useful for creative and diverse generation

Implementation Notes:
1. Helper Methods:
   - _apply_repeat_penalty: Penalizes repeated tokens
   - _filter_logits: Applies temperature and filtering
   - post_process_sequence: Handles EOS token truncation

2. Generation Methods:
   - generate_greedy: Implements basic greedy decoding
   - generate_beam: Implements beam search
   - generate_sample: Implements filtered sampling

3. Each generation method should:
   - Handle proper input validation
   - Track sequence scores
   - Handle EOS token detection
   - Support early stopping
'''

class SequenceGenerator:
    """
    A class for generating sequences using various decoding strategies.
    Supports greedy search, beam search, and sampling with top-k/nucleus filtering.
    """
    def __init__(
            self,
            score_fn: Callable,
            tokenizer: H4Tokenizer,
            max_length: int,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the sequence generator.
        
        Args:
            score_fn: Function that returns logits for next token prediction
            tokenizer: Tokenizer instance for handling token conversions
            max_length: Maximum sequence length to generate
            device: Device to run generation on
        """
        self.score_fn = score_fn
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def _apply_repeat_penalty(
            self,
            logits: torch.Tensor,
            sequences: torch.Tensor,
            penalty: float = 1.0
    ) -> torch.Tensor:
        """
        Apply repetition penalty to logits based on tokens in sequences.
        Args:
            logits: Logits tensor of shape (batch_size, vocab_size) or (batch_size, beam_width, vocab_size)
            sequences: Sequences tensor of shape (batch_size, sequence_length) or (batch_size, beam_width, sequence_length)
            penalty: Repetition penalty value
        Returns:
            Logits tensor with repetition penalty applied
        """
        if penalty == 1.0:
            return logits
        
        # Handle both regular and beam search shapes
        if logits.dim() == 2:
            # Greedy search: (batch_size, vocab_size)
            for idx in range(sequences.size(0)):
                unique_tokens = torch.unique(sequences[idx])
                logits[idx, unique_tokens] = logits[idx, unique_tokens] / torch.where(
                    logits[idx, unique_tokens] > 0,
                    torch.full_like(logits[idx, unique_tokens], penalty),
                    torch.full_like(logits[idx, unique_tokens], 1.0/penalty)
                )
        else:
            # Beam search: (batch_size, beam_width, vocab_size)
            for batch_idx in range(sequences.size(0)):
                for beam_idx in range(sequences.size(1)):
                    unique_tokens = torch.unique(sequences[batch_idx, beam_idx])
                    logits[batch_idx, beam_idx, unique_tokens] = logits[batch_idx, beam_idx, unique_tokens] / torch.where(
                        logits[batch_idx, beam_idx, unique_tokens] > 0,
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], penalty),
                        torch.full_like(logits[batch_idx, beam_idx, unique_tokens], 1.0/penalty)
                    )
        
        return logits

    def _filter_logits(
            self,
            logits: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature

        if top_k > 0:
            top_k_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            indices_to_remove = logits < top_k_logits[..., -1:]
            logits[indices_to_remove] = float('-inf')

        if top_p < 1.0:
            log_probs = torch.log_softmax(logits, dim=-1)
            sorted_log_probs, sorted_indices = torch.sort(log_probs, descending=True)
            cumulative_probs = torch.cumsum(torch.exp(sorted_log_probs), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        return logits

    def generate_greedy(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using greedy search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        batch_size = x.size(0)
        device = x.device

        scores = torch.zeros(batch_size, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break
            
            next_token_logits = self.score_fn(x)

            if repeat_penalty != 1.0:
                next_token_logits = self._apply_repeat_penalty(next_token_logits, x, repeat_penalty)

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            # Greedy selection - take the token with the highest probability
            next_tokens = torch.argmax(next_token_probs, dim=-1)  # (batch_size,)
            
            # Calculate log probabilities for score tracking
            log_probs = torch.log_softmax(next_token_logits, dim=-1)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1)  # (batch_size,)
            
            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)
            
            # Update finished flag for sequences that generated EOS token
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos
            
            # Create a mask for sequences that were just completed in this step
            just_finished = is_eos & ~finished
            
            # For sequences that just finished, replace the next token with EOS
            # to ensure proper token generation
            if just_finished.any():
                next_tokens = torch.where(just_finished, self.tokenizer.eos_id, next_tokens)
            
            # For already finished sequences, don't add new tokens (just repeat the last token)
            for i in range(batch_size):
                if finished[i] and not just_finished[i]:
                    next_tokens[i] = x[i, -1]
            
            # Append next tokens to the sequences
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1)
        
        return x, scores

    def generate_beam(
            self,
            x: torch.Tensor,
            beam_width: int,
            temperature: float = 1.0,
            repeat_penalty: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using beam search.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            beam_width: Number of beams to use
            temperature: Temperature for logits scaling
            repeat_penalty: Penalty for repeated tokens
        Returns:
            Tuple of tensors: (sequences, scores)
            - sequences is of shape (batch_size, beam_width, sequence_length) where each sequence in a beam set is sorted by score
            - scores is of shape (batch_size, beam_width)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        
        # Special case: if beam_width is 1, use greedy search
        if beam_width == 1:
            sequences, scores = self.generate_greedy(x, temperature, repeat_penalty)
            return sequences.unsqueeze(1), scores.unsqueeze(1)
        
        # Initialize variables
        batch_size = x.size(0)
        seq_len = x.size(1)
        device = x.device
        vocab_size = self.tokenizer.vocab_size
        
        # Create tensors to track beam search state
        # Start with a single beam per batch item (the input sequence)
        beam_scores = torch.zeros((batch_size, 1), device=device)
        beam_sequences = x.unsqueeze(1)  # (batch_size, 1, seq_len)
        
        # Track which beams have finished
        beam_finished = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        
        # Generate tokens until max_length is reached or all beams have finished
        for step in range(self.max_length - seq_len):
            # Check if all beams for all batch items have finished
            if beam_finished.all():
                break
            
            # Get the number of active beams for each batch item
            curr_beam_width = beam_sequences.size(1)
            
            # Reshape beam_sequences to feed into model
            # (batch_size * curr_beam_width, curr_seq_len)
            flat_sequences = beam_sequences.view(-1, beam_sequences.size(-1))
            
            # Get next token logits for all beams
            next_token_logits = self.score_fn(flat_sequences)  # (batch_size * curr_beam_width, vocab_size)
            
            # Reshape back to (batch_size, curr_beam_width, vocab_size)
            next_token_logits = next_token_logits.view(batch_size, curr_beam_width, -1)
            
            # Apply repetition penalty if needed
            if repeat_penalty != 1.0:
                next_token_logits = self._apply_repeat_penalty(next_token_logits, beam_sequences, repeat_penalty)
            
            # Apply temperature scaling
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Apply log softmax to convert to log probabilities
            vocab_log_probs = torch.log_softmax(next_token_logits, dim=-1)  # (batch_size, curr_beam_width, vocab_size)
            
            # Calculate scores for all possible next tokens for all beams
            # (batch_size, curr_beam_width, vocab_size)
            next_scores = beam_scores.unsqueeze(-1) + vocab_log_probs
            
            # For finished beams, only allow EOS token to have score 0, rest -inf
            mask = beam_finished.unsqueeze(-1).expand(batch_size, curr_beam_width, vocab_size)
            eos_mask = torch.zeros_like(mask)
            eos_mask[:, :, self.tokenizer.eos_id] = 1
            # Where beam is finished and token is not EOS, score = -inf
            next_scores = torch.where(mask & ~eos_mask, torch.full_like(next_scores, float('-inf')), next_scores)
            # Where beam is finished and token is EOS, score = beam_score
            next_scores = torch.where(mask & eos_mask, beam_scores.unsqueeze(-1), next_scores)
            
            # Flatten scores across beams for each batch item to prepare for top-k
            # (batch_size, curr_beam_width * vocab_size)
            flat_next_scores = next_scores.view(batch_size, -1)
            
            # If step is 0, we need to handle differently as we're expanding from 1 beam to beam_width
            if step == 0 and curr_beam_width == 1:
                # Select the top beam_width scoring tokens for each batch item
                # Note: In the first step, all candidate beams come from the same parent
                topk_scores, topk_indices = torch.topk(flat_next_scores, beam_width, dim=1)
                
                # Calculate beam indices and token indices
                topk_beam_indices = torch.zeros_like(topk_indices)  # All from beam 0
                topk_token_indices = topk_indices % vocab_size
            else:
                # For subsequent steps, we need to choose top-k from all candidates
                # Number of candidates to keep (might be less than beam_width if some beams finished)
                num_candidates = min(beam_width, curr_beam_width * vocab_size)
                
                # Select the top num_candidates scoring tokens for each batch item
                topk_scores, topk_indices = torch.topk(flat_next_scores, num_candidates, dim=1)
                
                # Calculate which beam each top-k candidate came from and the token
                topk_beam_indices = topk_indices // vocab_size
                topk_token_indices = topk_indices % vocab_size
            
            # Create new sequences by appending the selected tokens to their parent beams
            new_sequences = []
            new_scores = []
            new_finished = []
            
            for batch_idx in range(batch_size):
                batch_new_sequences = []
                batch_new_scores = []
                batch_new_finished = []
                
                for beam_idx in range(len(topk_scores[batch_idx])):
                    # Skip if score is -inf (happens for finished beams that tried to generate non-EOS tokens)
                    if topk_scores[batch_idx][beam_idx] == float('-inf'):
                        continue
                    
                    # Get the parent beam and token to append
                    parent_beam_idx = topk_beam_indices[batch_idx][beam_idx]
                    token_idx = topk_token_indices[batch_idx][beam_idx]
                    parent_sequence = beam_sequences[batch_idx, parent_beam_idx].clone()
                    
                    # Create the new sequence
                    new_sequence = torch.cat([parent_sequence, token_idx.unsqueeze(0)], dim=0)
                    batch_new_sequences.append(new_sequence)
                    
                    # Update the score
                    batch_new_scores.append(topk_scores[batch_idx][beam_idx])
                    
                    # Check if this beam has finished (generated EOS token)
                    new_finished_flag = beam_finished[batch_idx, parent_beam_idx] | (token_idx == self.tokenizer.eos_id)
                    batch_new_finished.append(new_finished_flag)
                    
                    # If we have enough beams, stop
                    if len(batch_new_sequences) >= beam_width:
                        break
                
                # Pad with copies of the first sequence if we don't have enough beams
                # This can happen if many beams have finished and generated -inf scores
                while len(batch_new_sequences) < beam_width:
                    # Use the best beam as a filler if we don't have enough valid beams
                    batch_new_sequences.append(batch_new_sequences[0].clone())
                    batch_new_scores.append(batch_new_scores[0])
                    batch_new_finished.append(batch_new_finished[0])
                
                # Stack the new sequences for this batch item
                new_sequences.append(torch.stack(batch_new_sequences))
                new_scores.append(torch.tensor(batch_new_scores, device=device))
                new_finished.append(torch.tensor(batch_new_finished, device=device))
            
            # Update the beam state
            beam_sequences = torch.stack(new_sequences)  # (batch_size, beam_width, new_seq_len)
            beam_scores = torch.stack(new_scores)        # (batch_size, beam_width)
            beam_finished = torch.stack(new_finished)    # (batch_size, beam_width)
        
        # Ensure all sequences have the same length (pad if necessary)
        # This can happen if we terminated early due to all beams finishing
        if beam_sequences.size(2) < self.max_length:
            padding_length = self.max_length - beam_sequences.size(2)
            padding = torch.full((batch_size, beam_width, padding_length), 
                                self.tokenizer.pad_id, 
                                device=device)
            beam_sequences = torch.cat([beam_sequences, padding], dim=2)
        
        # Return the final sequences and scores
        return beam_sequences, beam_scores

    def generate_sample(
            self,
            x: torch.Tensor,
            temperature: float = 1.0,
            top_k: int = 0,
            top_p: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences using sampling with top-k and nucleus filtering.
        Args:
            x: Input tensor of shape (batch_size, sequence_length)
            temperature: Temperature for logits scaling
            top_k: Number of top-k tokens to sample from
            top_p: Proportion of top-p tokens to sample from
        Returns:
            Tuple of tensors: (sequences, scores)
             - sequences is of shape (batch_size, sequence_length)
             - scores is of shape (batch_size,)
        """
        # Add input validation
        if not torch.is_tensor(x):
            raise TypeError("Input x must be a torch tensor")
        if x.dim() != 2:
            raise ValueError("Input x must be 2-dimensional (batch_size, seq_len)")
        if self.max_length < x.size(1):
            raise ValueError("max_length must be >= input sequence length")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < top_p <= 1.0:
            raise ValueError("top_p must be > 0 and <= 1.0")
        
        # Initialize scores and finished flag
        batch_size = x.size(0)
        scores = torch.zeros(batch_size, device=x.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=x.device)

        for _ in range(self.max_length - x.size(1)):
            # Check if all sequences have finished
            if finished.all():
                break

            # Get logits and apply filtering
            next_scores = self.score_fn(x) # (batch_size, vocab_size)
            filtered_logits = self._filter_logits(next_scores, temperature, top_k, top_p)
            log_probs = torch.log_softmax(filtered_logits, dim=-1)
            
            # We need probabilities for multinomial sampling
            probs = torch.exp(log_probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1) # (batch_size,)
            token_scores = log_probs.gather(1, next_tokens.unsqueeze(1)).squeeze(1) # (batch_size,)

            # Update scores only for unfinished sequences
            scores = torch.where(finished, scores, scores + token_scores)

            # Append next tokens
            x = torch.cat([x, next_tokens.unsqueeze(1)], dim=1) # (batch_size, seq_len + 1)

            # Check if any sequence has reached EOS 
            is_eos = (next_tokens == self.tokenizer.eos_id)
            finished = finished | is_eos

        return x, scores

    @staticmethod
    def post_process_sequence(seq: torch.Tensor, tokenizer: H4Tokenizer) -> torch.Tensor:
        """
        Post process sequences to remove content after EOS token.
        Args:
            seq: Input tensor of shape (batch_size, sequence_length) or (sequence_length)
            tokenizer: Tokenizer instance for handling token conversions
        Returns:
            if seq is a single sequence, return a tensor of same shape with sequence truncated at EOS
            if seq is a batch of sequences, return a list of tensors with each sequence truncated at first EOS
        """
        # Handle single sequence case
        if seq.dim() == 1:
            eos_indices = (seq == tokenizer.eos_id).nonzero()
            if len(eos_indices) > 0:
                end_idx = eos_indices[0].item() + 1
                return seq[:end_idx]
            return seq
        
        # Handle batched sequences
        eos_mask = seq == tokenizer.eos_id  # (batch_size, sequence_length)
        # Find first EOS token in each sequence
        eos_indices = eos_mask.float().cumsum(dim=1).eq(1) & eos_mask
        # Create sequence mask that includes everything up to and including first EOS
        seq_mask = eos_indices.cumsum(dim=1).eq(0) | eos_indices
        # Apply mask and pack sequences
        return [s[:m.sum()] for s, m in zip(seq, seq_mask)]