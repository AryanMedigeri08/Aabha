"""
================================================================================
FILE 4: MULTI-HEAD SELF-ATTENTION — The Heart of the Transformer
================================================================================

WHAT THIS MODULE DOES:
    Implements Multi-Head Self-Attention (MHSA) from scratch. This is THE
    mechanism that gives Transformers their power — it allows every token in
    the sequence to "look at" every other token and decide what's important.

    In a Vision Transformer, this means every image patch can attend to every
    other patch, enabling the model to learn long-range spatial relationships
    that CNNs struggle with (e.g., "the wheel at the bottom-left is connected
    to the car body at the center").

WHY "SELF" ATTENTION?
    "Self" means the Queries, Keys, and Values all come from the SAME input
    sequence. Each token generates its own Q, K, and V vectors.

    (In cross-attention, Q comes from one sequence and K,V come from another.
    We'll see that later when we evolve toward BLIP.)

THE Q-K-V INTUITION:
    Think of attention like a library search:

    Query (Q):  "What am I looking for?"
        Each token generates a query vector that encodes what information
        it NEEDS from other tokens.

    Key (K):    "What do I contain?"
        Each token generates a key vector that encodes what information
        it HAS to offer.

    Value (V):  "Here's my actual content."
        Each token generates a value vector that contains the actual
        information to be passed along if selected.

    The attention mechanism:
        1. Compare each Q against all Ks (dot product) -- "How relevant is
           each token to what I'm looking for?"
        2. Normalize scores with softmax -- "Convert relevances to a
           probability distribution."
        3. Use these scores to compute a weighted sum of Vs -- "Aggregate
           the relevant information."

MATHEMATICAL FORMULATION:
    Given input X of shape (B, N, D):

    Step 1: Linear Projections
        Q = X @ W_q + b_q    # shape: (B, N, D)
        K = X @ W_k + b_k    # shape: (B, N, D)
        V = X @ W_v + b_v    # shape: (B, N, D)

    Step 2: Split into Multiple Heads
        Reshape Q, K, V from (B, N, D) to (B, num_heads, N, head_dim)
        where head_dim = D / num_heads

        Why multiple heads?
            A single attention head can only learn ONE type of relationship.
            Multiple heads let the model simultaneously attend to different
            types of relationships:
                Head 1 might learn "spatial proximity" (nearby patches)
                Head 2 might learn "color similarity"
                Head 3 might learn "edge continuation"
                etc.

            With 12 heads and D=768: each head operates on head_dim = 64.
            This is cheaper than one giant head of size 768, and more expressive
            because we get 12 independent attention patterns.

    Step 3: Scaled Dot-Product Attention (per head)
        scores = (Q @ K^T) / sqrt(head_dim)    # shape: (B, H, N, N)

        Why divide by sqrt(head_dim)?
            Without scaling, the dot products grow proportionally to head_dim.
            Large dot products push softmax into regions with tiny gradients
            (vanishing gradient problem). Dividing by sqrt(head_dim) keeps the
            variance of the dot products at ~1, regardless of dimension.

            Proof: If q and k are vectors with entries drawn i.i.d. from
            N(0,1), then E[q . k] = 0 and Var[q . k] = head_dim.
            So std(q . k) = sqrt(head_dim).
            Dividing by sqrt(head_dim) normalizes: Var(q . k / sqrt(d)) = 1.

        attn_weights = softmax(scores, dim=-1)   # shape: (B, H, N, N)

        The attention weight matrix is (N x N): entry [i, j] tells us how
        much token i should attend to token j. Each row sums to 1.

        output = attn_weights @ V                 # shape: (B, H, N, head_dim)

    Step 4: Concatenate Heads and Project
        Reshape output from (B, H, N, head_dim) back to (B, N, D)
        Apply final linear projection: output = output @ W_o + b_o

        This final projection lets the model learn how to combine information
        from different heads.

SHAPE TRANSFORMATIONS (B=2, N=197, D=768, H=12, head_dim=64):
    Input:                   (2, 197, 768)
    After Q/K/V projection:  (2, 197, 768) each
    After reshape+transpose: (2, 12, 197, 64) each
    Attention scores:        (2, 12, 197, 197)  -- N x N attention map!
    Attention weights:       (2, 12, 197, 197)  -- after softmax
    Attention output:        (2, 12, 197, 64)
    After concat:            (2, 197, 768)      -- heads reassembled
    After output projection: (2, 197, 768)      -- final output

PARAMETER COUNT:
    W_q: D x D = 768 x 768 = 589,824
    W_k: D x D = 768 x 768 = 589,824
    W_v: D x D = 768 x 768 = 589,824
    W_o: D x D = 768 x 768 = 589,824
    Biases: 4 x 768 = 3,072
    Total: 2,362,368 parameters per attention layer

CONNECTION TO BLIP:
    BLIP's visual encoder uses the same MHSA mechanism. Later, BLIP also
    uses CROSS-attention where Q comes from text tokens and K,V come from
    image tokens. The core mechanism is identical — only the source of
    Q, K, V changes.
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention mechanism built from scratch.

    Every operation is explicit — no hidden abstractions.

    Args:
        embed_dim: Total embedding dimension D (e.g., 768).
        num_heads: Number of attention heads H (e.g., 12).
        attn_drop: Dropout rate applied to attention weights.
        proj_drop: Dropout rate applied to the output projection.

    Example:
        >>> mhsa = MultiHeadSelfAttention(embed_dim=768, num_heads=12)
        >>> x = torch.randn(2, 197, 768)   # (batch, seq_len, embed_dim)
        >>> output = mhsa(x)                # shape: (2, 197, 768)
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()

        # ── Validate that embed_dim is divisible by num_heads ──
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads "
            f"({num_heads}). Got remainder: {embed_dim % num_heads}."
        )

        self.embed_dim: int = embed_dim
        self.num_heads: int = num_heads
        self.head_dim: int = embed_dim // num_heads
        # Scale factor for dot-product attention: 1 / sqrt(head_dim)
        self.scale: float = self.head_dim ** -0.5

        # =====================================================================
        # Q, K, V Projection Matrices
        # =====================================================================
        # Each of these is a linear layer that projects from D to D.
        # Internally, you can think of each as projecting from D to
        # (num_heads * head_dim), which equals D.
        #
        # We use SEPARATE linear layers for Q, K, V to make the computation
        # explicit and educational. In production code (like in BLIP), you'll
        # often see a SINGLE linear layer that projects to 3*D and then splits
        # the output into Q, K, V — this is slightly faster due to a single
        # matrix multiply, but harder to understand.
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # =====================================================================
        # Output Projection
        # =====================================================================
        # After concatenating all head outputs back together, we apply one
        # final linear projection. This lets the model learn how to combine
        # information from different heads.
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # =====================================================================
        # Dropout Layers
        # =====================================================================
        # Attention dropout: randomly zeros out attention weights during
        # training. This prevents the model from relying too heavily on
        # specific token-to-token relationships, encouraging it to learn
        # more robust, distributed attention patterns.
        self.attn_dropout = nn.Dropout(attn_drop)

        # Projection dropout: applied after the output projection.
        self.proj_dropout = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        assert D == self.embed_dim, (
            f"Expected embed_dim={self.embed_dim}, got {D}."
        )

        print(f"\n{'='*60}")
        print(f"MULTI-HEAD SELF-ATTENTION -- Forward Pass")
        print(f"{'='*60}")
        print(f"Config: num_heads={self.num_heads}, head_dim={self.head_dim}, "
              f"scale={self.scale:.6f}")
        print(f"Input shape: {x.shape}")
        print(f"  => (batch={B}, seq_len={N}, embed_dim={D})")

        # =====================================================================
        # STEP 1: Compute Q, K, V via linear projections
        # =====================================================================
        # Each projection: (B, N, D) @ (D, D) + bias = (B, N, D)
        # Conceptually:
        #   Q[i] = "What is token i looking for?"
        #   K[i] = "What does token i contain?"
        #   V[i] = "What information does token i carry?"
        Q = self.W_q(x)   # (B, N, D)
        K = self.W_k(x)   # (B, N, D)
        V = self.W_v(x)   # (B, N, D)

        print(f"\nStep 1 -- Q, K, V Projections:")
        print(f"  Q shape: {Q.shape}")
        print(f"  K shape: {K.shape}")
        print(f"  V shape: {V.shape}")
        print(f"  (Each is input @ W + b, shape preserved at (B, N, D))")

        # =====================================================================
        # STEP 2: Reshape into multiple heads
        # =====================================================================
        # Current shape: (B, N, D) where D = num_heads * head_dim
        #
        # We need to split the last dimension into (num_heads, head_dim):
        #   (B, N, D) -> (B, N, num_heads, head_dim)
        #
        # Then transpose to bring heads before the sequence dimension:
        #   (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        #
        # Why this order? We want each head to independently attend over
        # the full sequence. With shape (B, H, N, d_k), the matrix multiply
        # Q @ K^T operates on the last two dims: (N, d_k) @ (d_k, N) = (N, N),
        # which is exactly the attention matrix we want.

        Q = Q.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # Each is now (B, num_heads, N, head_dim) = (2, 12, 197, 64)

        print(f"\nStep 2 -- Reshape into {self.num_heads} heads:")
        print(f"  Q shape: {Q.shape}")
        print(f"  K shape: {K.shape}")
        print(f"  V shape: {V.shape}")
        print(f"  (B, num_heads, seq_len, head_dim)")

        # =====================================================================
        # STEP 3: Scaled Dot-Product Attention
        # =====================================================================
        # This is the core attention computation.
        #
        # 3a. Compute raw attention scores: Q @ K^T
        #     For each head: (N, head_dim) @ (head_dim, N) = (N, N)
        #     Full shape: (B, H, N, N)
        #     Entry [b, h, i, j] = how much token i attends to token j
        #                           in head h of sample b.

        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        # K.transpose(-2, -1) swaps the last two dims:
        #   (B, H, N, head_dim) -> (B, H, head_dim, N)
        # matmul: (B, H, N, head_dim) @ (B, H, head_dim, N) = (B, H, N, N)

        print(f"\nStep 3a -- Raw attention scores (Q @ K^T):")
        print(f"  Shape: {attn_scores.shape}")
        print(f"  => (B, H, N, N) = ({B}, {self.num_heads}, {N}, {N})")
        print(f"  This is a {N}x{N} attention map per head per sample!")
        print(f"  Total attention scores: {B} * {self.num_heads} * {N} * {N} "
              f"= {B * self.num_heads * N * N:,}")

        # 3b. Scale by 1/sqrt(head_dim)
        #     Without this, dot products grow with head_dim, pushing softmax
        #     into saturation (near-zero gradients).
        #
        #     Mathematical justification:
        #     If q_i, k_i ~ N(0, 1), then q . k = sum(q_i * k_i) has:
        #       E[q . k] = 0
        #       Var[q . k] = head_dim
        #     Dividing by sqrt(head_dim) gives Var = 1, keeping softmax
        #     in a well-behaved regime.

        attn_scores = attn_scores * self.scale

        print(f"\nStep 3b -- After scaling by 1/sqrt({self.head_dim}) "
              f"= {self.scale:.6f}:")
        print(f"  Score stats: min={attn_scores.min().item():.4f}, "
              f"max={attn_scores.max().item():.4f}, "
              f"mean={attn_scores.mean().item():.4f}")

        # 3c. Apply softmax along the last dimension (over keys)
        #     This converts raw scores into a probability distribution.
        #     For each query token i, the attention weights over all key
        #     tokens j sum to 1.
        #     attn_weights[b, h, i, :] is a probability distribution over
        #     all N tokens, telling us how much token i should "pay attention"
        #     to each token j.

        attn_weights = F.softmax(attn_scores, dim=-1)

        print(f"\nStep 3c -- After softmax (attention weights):")
        print(f"  Shape: {attn_weights.shape}")
        print(f"  Row sum (should be ~1.0): "
              f"{attn_weights[0, 0, 0, :].sum().item():.6f}")
        print(f"  Weight stats: min={attn_weights.min().item():.6f}, "
              f"max={attn_weights.max().item():.6f}")

        # 3d. Apply attention dropout (only during training)
        #     Randomly zeros out some attention weights to prevent the model
        #     from forming overly rigid attention patterns.
        attn_weights = self.attn_dropout(attn_weights)

        # 3e. Compute the weighted sum of values
        #     For each query token i, we compute a weighted combination of
        #     ALL value vectors, where the weights come from the attention
        #     distribution computed above.
        #
        #     (B, H, N, N) @ (B, H, N, head_dim) = (B, H, N, head_dim)
        #
        #     Intuitively: each output token is a "mixture" of all input
        #     tokens' values, mixed according to relevance (attention weights).

        attn_output = torch.matmul(attn_weights, V)

        print(f"\nStep 3e -- Attention output (weights @ V):")
        print(f"  Shape: {attn_output.shape}")
        print(f"  => (B, num_heads, seq_len, head_dim)")

        # =====================================================================
        # STEP 4: Concatenate heads and apply output projection
        # =====================================================================
        # We need to reverse the reshape from Step 2:
        #   (B, H, N, head_dim) -> (B, N, H, head_dim) -> (B, N, D)
        #
        # transpose(1, 2): swap H and N dims back
        # reshape: merge (H, head_dim) back into D
        #
        # .contiguous() is needed because transpose() returns a view with
        # non-contiguous memory layout, and reshape() requires contiguous data.

        attn_output = attn_output.transpose(1, 2).contiguous()
        # Now: (B, N, num_heads, head_dim)

        attn_output = attn_output.reshape(B, N, self.embed_dim)
        # Now: (B, N, D) -- heads are concatenated back together

        print(f"\nStep 4a -- After concat heads:")
        print(f"  Shape: {attn_output.shape}")
        print(f"  (All {self.num_heads} heads merged back: "
              f"{self.num_heads} x {self.head_dim} = {self.embed_dim})")

        # Apply the final output projection
        # This linear layer lets the model learn how to combine information
        # that was independently computed by each head.
        output = self.W_o(attn_output)
        output = self.proj_dropout(output)

        print(f"\nStep 4b -- After output projection + dropout:")
        print(f"  Shape: {output.shape}")
        print(f"  => (batch={B}, seq_len={N}, embed_dim={self.embed_dim})")
        print(f"{'='*60}\n")

        return output


# =============================================================================
# STANDALONE TEST
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING MULTI-HEAD SELF-ATTENTION MODULE")
    print("=" * 60)

    # Configuration (ViT-Base defaults)
    EMBED_DIM = 768
    NUM_HEADS = 12
    SEQ_LEN = 197      # 196 patches + 1 CLS token
    BATCH_SIZE = 2

    # Create the MHSA layer
    mhsa = MultiHeadSelfAttention(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        attn_drop=0.0,
        proj_drop=0.0,
    )

    # Create dummy input (as if coming from patch embedding + CLS + pos embed)
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    print(f"\nCreated dummy input of shape: {dummy_input.shape}")

    # Run forward pass
    output = mhsa(dummy_input)

    # ── Verify output shape ──
    expected_shape = (BATCH_SIZE, SEQ_LEN, EMBED_DIM)
    assert output.shape == expected_shape, (
        f"Shape mismatch! Expected {expected_shape}, got {output.shape}"
    )
    print(f"[OK] Output shape verified: {output.shape}")

    # ── Verify that output is different from input ──
    # (attention should transform the representations)
    assert not torch.allclose(output, dummy_input, atol=1e-6), (
        "Output is identical to input -- attention did nothing!"
    )
    print(f"[OK] Output differs from input (attention transformed the data)")

    # ── Parameter count ──
    print(f"\nModule Parameters:")
    total_params = 0
    for name, param in mhsa.named_parameters():
        num_params = param.numel()
        total_params += num_params
        print(f"  {name}: {list(param.shape)} = {num_params:,} params")
    print(f"  {'---'*10}")
    print(f"  Total: {total_params:,} parameters")

    # ── Verify parameter count ──
    # 4 linear layers (Q, K, V, O), each with D*D weights + D bias
    expected_params = 4 * (EMBED_DIM * EMBED_DIM + EMBED_DIM)
    assert total_params == expected_params, (
        f"Parameter count mismatch! Expected {expected_params:,}, "
        f"got {total_params:,}"
    )
    print(f"[OK] Parameter count verified: {total_params:,} "
          f"(expected {expected_params:,})")

    # ── Verify head_dim computation ──
    assert mhsa.head_dim == 64, f"Expected head_dim=64, got {mhsa.head_dim}"
    print(f"[OK] head_dim = {mhsa.head_dim} "
          f"({EMBED_DIM} / {NUM_HEADS} = {EMBED_DIM // NUM_HEADS})")

    print(f"\n{'='*60}")
    print(f"ALL TESTS PASSED!")
    print(f"{'='*60}")
