"""
================================================================================
FILE 3: CLS TOKEN — The Global Summary Token
================================================================================

WHAT THIS MODULE DOES:
    Prepends a single learnable [CLS] (classification) token to the beginning
    of a sequence of patch embeddings. This token does NOT correspond to any
    image patch — it is a "blank slate" that learns to aggregate information
    from all patches through self-attention.

WHY IT EXISTS:
    After passing through multiple Transformer encoder blocks, each token
    attends to every other token. The CLS token, having no inherent content
    of its own, becomes a natural "summary" of the entire image.

    At the end of the Transformer, we extract ONLY the CLS token's output
    and feed it into a classification head. This avoids needing to pool or
    average across all 196 patch tokens.

    Historical context:
        This design originates from BERT (Devlin et al., 2018) in NLP, where
        a [CLS] token is prepended to the input sentence for classification.
        The ViT paper (Dosovitskiy et al., 2020) adopted the same idea for
        images: prepend a learnable CLS token to the patch sequence.

MATHEMATICAL INTUITION:
    Let X ∈ R^(B, N, D) be the patch embeddings where:
        B = batch size
        N = number of patches (e.g., 196)
        D = embedding dimension (e.g., 768)

    The CLS token is a learnable parameter:
        cls_token ∈ R^(1, 1, D)

    We expand it across the batch dimension:
        cls_tokens = cls_token.expand(B, 1, D)   # shape: (B, 1, D)

    Then concatenate it to the front of the patch sequence:
        output = concat([cls_tokens, X], dim=1)   # shape: (B, N+1, D)

    Why prepend and not append?
        Convention from BERT. The position doesn't fundamentally matter
        because self-attention is position-agnostic (that's why we add
        positional embeddings separately). But prepending means the CLS
        token is always at index 0, making extraction trivial: output[:, 0].

SHAPE TRANSFORMATIONS:
    Input patch embeddings:          (B, 196, 768)
    CLS token parameter:             (1,   1, 768)
    Expanded CLS tokens:             (B,   1, 768)
    Concatenated output:             (B, 197, 768)
                                          ^
                                          N + 1 (196 patches + 1 CLS)

COMPLEXITY ANALYSIS:
    Time: O(B * D) for the expand + O(B * (N+1) * D) for the concatenation.
    In practice, this is negligible compared to attention.

    Space (Parameters): O(D) = 768 parameters.
    This is tiny — just one vector.

COMMON MISTAKES:
    1. Using cls_token.repeat(B, 1, 1) instead of cls_token.expand(B, 1, 1).
       - repeat() copies data in memory, allocating B * D floats.
       - expand() creates a view with shared memory — zero copy, zero allocation.
       - Both produce the same output tensor shape, but expand() is much more
         memory efficient. Always prefer expand() for broadcasting.
    2. Forgetting to account for the extra token when computing positional
       embeddings. If you have 196 patches, the positional embedding must
       have length 197 (196 + 1 for CLS).
    3. Concatenating along the wrong dimension (dim=0 or dim=2 instead of dim=1).
       dim=1 is the sequence dimension.
================================================================================
"""

import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """Prepends a learnable [CLS] classification token to a sequence.

    The CLS token is a single learnable vector of size embed_dim. During the
    forward pass, it is expanded to match the batch size and concatenated
    to the front of the input token sequence.

    Args:
        embed_dim: The dimension of the token embeddings (D).

    Example:
        >>> cls = CLSToken(embed_dim=768)
        >>> patches = torch.randn(2, 196, 768)   # 196 patch embeddings
        >>> output = cls(patches)                  # shape: (2, 197, 768)
        >>> cls_output = output[:, 0]              # extract CLS: (2, 768)
    """

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()

        self.embed_dim: int = embed_dim

        # ── Learnable CLS Token ──
        # Shape (1, 1, D): first dim is "batch" placeholder for broadcasting,
        # second dim is the single token in the sequence, third is embedding dim.
        #
        # Initialized with truncated normal (std=0.02) following the ViT paper,
        # which inherits this from BERT's initialization strategy.
        # Small initial values prevent the CLS token from dominating early
        # attention computations before the network has learned anything useful.
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, self.embed_dim) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend the CLS token to the input sequence.

        Args:
            x: Patch embeddings of shape (B, N, D).

        Returns:
            Sequence with CLS token prepended, shape (B, N+1, D).
        """
        batch_size, num_patches, embed_dim = x.shape

        assert embed_dim == self.embed_dim, (
            f"Expected embed_dim={self.embed_dim}, but got {embed_dim}."
        )

        print(f"\n{'='*60}")
        print(f"CLS TOKEN -- Forward Pass")
        print(f"{'='*60}")
        print(f"Input shape: {x.shape}")
        print(f"  => (batch={batch_size}, num_patches={num_patches}, "
              f"embed_dim={embed_dim})")
        print(f"CLS token parameter shape: {self.cls_token.shape}")

        # =====================================================================
        # STEP 1: Expand the CLS token across the batch dimension
        # =====================================================================
        # self.cls_token has shape (1, 1, D).
        # We need one copy per sample in the batch: (B, 1, D).
        #
        # expand() creates a VIEW — it does NOT allocate new memory.
        # The underlying data is shared; PyTorch just adjusts the strides
        # so that reading index [i, 0, :] returns the same data for all i.
        #
        # Why not repeat()? repeat() physically copies the data B times,
        # wasting memory. expand() achieves the same result for free.
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # -1 means "don't change this dimension's size"
        # Result shape: (B, 1, D)

        print(f"\nAfter expand to batch: {cls_tokens.shape}")
        print(f"  => (batch={batch_size}, 1, embed_dim={embed_dim})")

        # =====================================================================
        # STEP 2: Concatenate CLS token to the front of the patch sequence
        # =====================================================================
        # We concatenate along dim=1 (the sequence dimension).
        # [CLS, patch_1, patch_2, ..., patch_196]
        #
        # After this, the sequence length increases from N to N+1.
        # The CLS token is always at index 0 in the sequence.
        output = torch.cat([cls_tokens, x], dim=1)

        new_seq_len = num_patches + 1
        print(f"\nAfter concatenation: {output.shape}")
        print(f"  => (batch={batch_size}, seq_len={new_seq_len}, "
              f"embed_dim={embed_dim})")
        print(f"  CLS token is at index 0: output[:, 0, :] has shape "
              f"{output[:, 0, :].shape}")
        print(f"{'='*60}\n")

        return output


# =============================================================================
# STANDALONE TEST — Run this file directly to see the shapes in action
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING CLS TOKEN MODULE")
    print("=" * 60)

    # Configuration
    EMBED_DIM = 768
    NUM_PATCHES = 196
    BATCH_SIZE = 2

    # Create the CLS token layer
    cls_layer = CLSToken(embed_dim=EMBED_DIM)

    # Create dummy patch embeddings
    dummy_patches = torch.randn(BATCH_SIZE, NUM_PATCHES, EMBED_DIM)
    print(f"\nCreated dummy patches of shape: {dummy_patches.shape}")

    # Run forward pass
    output = cls_layer(dummy_patches)

    # Verify output shape: should be (B, N+1, D)
    expected_shape = (BATCH_SIZE, NUM_PATCHES + 1, EMBED_DIM)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )

    # Verify CLS token is at index 0
    cls_output = output[:, 0, :]
    assert cls_output.shape == (BATCH_SIZE, EMBED_DIM), (
        f"CLS output shape mismatch! Expected {(BATCH_SIZE, EMBED_DIM)}, "
        f"got {cls_output.shape}"
    )

    print("[OK] Output shape verified!")
    print(f"[OK] CLS token extraction verified! Shape: {cls_output.shape}")
    print(f"\nModule Parameters:")
    total_params = sum(p.numel() for p in cls_layer.parameters())
    print(f"  CLS token: {cls_layer.cls_token.shape} "
          f"= {cls_layer.cls_token.numel():,} params")
    print(f"  Total parameters: {total_params:,}")
