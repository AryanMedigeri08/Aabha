"""
================================================================================
FILE 2: LEARNABLE POSITIONAL EMBEDDING — Giving Spatial Context to Tokens
================================================================================

WHAT THIS MODULE DOES:
    Adds position-specific vectors (positional embeddings) to our patch embeddings.
    It takes an input sequence of tokens and adds a unique, learnable parameter
    vector to each token based on its index in the sequence.

WHY IT EXISTS:
    Transformers are permutation-invariant. The self-attention mechanism processes 
    all tokens in parallel and treats a sequence as an unordered set (bag of tokens).
    
    If you shuffle the order of patches in an image, the self-attention output
    would remain exactly the same (just shuffled), even though a scrambled image
    has completely different semantic meaning.
    
    To fix this, we must inject "positional information" into the network. 
    By adding a unique vector to each patch embedding that represents its position, 
    the transformer can learn to associate features with their locations in the image.

MATHEMATICAL INTUITION:
    Let X ∈ ℝ^(B × N × D) be the patch embeddings (where N is the sequence length, 
    e.g., 196 patches + 1 CLS token = 197, and D is the embedding dimension, e.g., 768).
    
    We define a learnable parameter matrix:
        E_pos ∈ ℝ^(1 × N × D)
        
    We inject positional information by simply adding E_pos to X:
        Y = X + E_pos
        
    Why addition and not concatenation?
    1. Concatenation would increase the feature dimension (D + N), making subsequent
       attention operations computationally more expensive.
    2. Addition preserves the feature dimension D.
    3. Since neural networks can learn to decompose linear combinations, the network 
       can easily learn to separate the "content" (from X) and the "location" (from E_pos).

SHAPE TRANSFORMATIONS:
    Input tokens (including CLS token): (B, N, D)
    ↓
    Positional Embeddings parameter:    (1, N, D)
    ↓ Broadcast addition along the batch dimension (B)
    Output tokens:                      (B, N, D)

COMPLEXITY ANALYSIS:
    Let B = batch_size, N = sequence_length, D = embed_dim
    
    Time Complexity: O(B · N · D)
        - The operation is a simple element-wise addition of two tensors.
        
    Space Complexity (Parameters): O(N · D)
        - We store one vector of size D for each position.
        - For N = 197, D = 768: 197 × 768 = 151,296 parameters.

COMMON MISTAKES:
    1. Not allowing the batch dimension to broadcast. If the parameter is initialized
       as shape (N, D) instead of (1, N, D), PyTorch won't be able to broadcast add it
       to a batch of shape (B, N, D) automatically.
    2. Initializing with values that are too large, which can overwhelm the initial 
       patch content. It is best to initialize them with a small standard deviation
       (e.g., 0.02) or zeros, allowing the network to adjust them during training.
    3. Shuffling or sorting the positional embeddings. The mapping between index
       i and the embedding E_pos[0, i] must remain strictly static.
================================================================================
"""

import torch
import torch.nn as nn


class LearnablePositionalEmbedding(nn.Module):
    """Applies learnable 1D positional embeddings to sequence embeddings.

    This layer adds a learnable parameter tensor of shape (1, seq_len, embed_dim)
    to the input tensor of shape (B, seq_len, embed_dim).

    Args:
        seq_len: The length of the sequence (num_patches + 1 for CLS token).
        embed_dim: The dimension of the token embeddings (D).

    Example:
        >>> pos_embed = LearnablePositionalEmbedding(seq_len=197, embed_dim=768)
        >>> x = torch.randn(2, 197, 768)  # Batch size of 2
        >>> out = pos_embed(x)             # Shape: (2, 197, 768)
    """

    def __init__(self, seq_len: int, embed_dim: int) -> None:
        super().__init__()
        
        self.seq_len: int = seq_len
        self.embed_dim: int = embed_dim

        # ── Learnable Parameter Definition ──
        # We define the positional embeddings as a learnable parameter.
        # Initializing shape as (1, seq_len, embed_dim) allows PyTorch's
        # broadcasting mechanism to automatically add it to any batch size (B, seq_len, embed_dim).
        #
        # We initialize it with a normal distribution with a small std (0.02),
        # which is standard for ViT (matching BERT initialization).
        self.pos_embeddings = nn.Parameter(
            torch.randn(1, self.seq_len, self.embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional embeddings to the input sequence tensor.

        Args:
            x: Input tensor of shape (B, seq_len, embed_dim).

        Returns:
            Tensor with positional context of shape (B, seq_len, embed_dim).
        """
        batch_size, seq_len, embed_dim = x.shape

        # ── Validate Input Shapes ──
        assert seq_len == self.seq_len, (
            f"Expected sequence length {self.seq_len}, but got {seq_len}."
        )
        assert embed_dim == self.embed_dim, (
            f"Expected embedding dimension {self.embed_dim}, but got {embed_dim}."
        )

        print(f"\n{'='*60}")
        print(f"POSITIONAL EMBEDDING -- Forward Pass")
        print(f"{'='*60}")
        print(f"Input shape: {x.shape}")
        print(f"  => (batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim})")
        print(f"Positional embedding parameter shape: {self.pos_embeddings.shape}")
        print(f"  => (1, seq_len={self.seq_len}, embed_dim={self.embed_dim})")

        # =====================================================================
        # STEP 1: Add the positional embeddings to the token sequence
        # =====================================================================
        # Due to PyTorch's broadcasting rules, the singleton dimension (1) 
        # in self.pos_embeddings is duplicated (conceptually, without memory copying)
        # B times to match the shape of x.
        #
        # Output shape: (B, N, D) + (1, N, D) -> (B, N, D)
        out = x + self.pos_embeddings
        
        print(f"\nAfter element-wise addition: {out.shape}")
        print(f"  => (batch={batch_size}, seq_len={seq_len}, embed_dim={embed_dim})")
        print(f"{'='*60}\n")

        return out


# =============================================================================
# STANDALONE TEST — Run this file directly to see the shapes in action
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING POSITIONAL EMBEDDING MODULE")
    print("=" * 60)

    # Configuration matching ViT-Base (196 patches + 1 CLS token)
    SEQ_LEN = 197
    EMBED_DIM = 768
    BATCH_SIZE = 2

    # Create the learnable positional embedding layer
    pos_embedding_layer = LearnablePositionalEmbedding(
        seq_len=SEQ_LEN,
        embed_dim=EMBED_DIM
    )

    # Create dummy tokens (representing patch + CLS embeddings)
    dummy_tokens = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"\nCreated dummy tokens of shape: {dummy_tokens.shape}")

    # Run forward pass
    output = pos_embedding_layer(dummy_tokens)

    # Verify output shape
    assert output.shape == (BATCH_SIZE, SEQ_LEN, EMBED_DIM), (
        f"Shape mismatch! Expected {(BATCH_SIZE, SEQ_LEN, EMBED_DIM)}, "
        f"got {output.shape}"
    )

    print("[OK] Output shape verified!")
    print(f"\nModule Parameters:")
    print(f"  Positional embeddings: {pos_embedding_layer.pos_embeddings.shape} "
          f"= {pos_embedding_layer.pos_embeddings.numel():,} params")
