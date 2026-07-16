"""
================================================================================
FILE 1: PATCH EMBEDDING — The Bridge Between Images and Transformers
================================================================================

WHAT THIS MODULE DOES:
    Converts a 2D image into a sequence of 1D embedding vectors.
    This is the critical first step that allows a Transformer (designed for
    sequences like text) to process images.

WHY IT EXISTS:
    Transformers operate on SEQUENCES of tokens (like words in a sentence).
    Images are 2D grids of pixels, not sequences. We need a way to convert
    an image into a sequence of "visual tokens."

    The solution from "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020):
    1. Split the image into fixed-size patches (like cutting a photo into tiles).
    2. Flatten each patch into a 1D vector.
    3. Linearly project each flattened patch into an embedding space.

    Each patch becomes one "token" — analogous to a word in NLP.

MATHEMATICAL INTUITION:
    Given an image of shape (C, H, W) = (3, 224, 224):

    Step 1 — Divide into patches:
        Patch size P = 16
        Number of patches along height: H / P = 224 / 16 = 14
        Number of patches along width:  W / P = 224 / 16 = 14
        Total patches: N = 14 × 14 = 196

    Step 2 — Flatten each patch:
        Each patch has shape (C, P, P) = (3, 16, 16)
        Flattened: 3 × 16 × 16 = 768 values per patch

    Step 3 — Linear projection:
        Project each 768-dim flattened patch → D-dim embedding (e.g., D = 768)
        This is a learnable linear layer: W ∈ ℝ^(768 × D), b ∈ ℝ^D
        output = flatten(patch) @ W + b

    Final output shape: (batch_size, 196, D)
    → A sequence of 196 tokens, each with D dimensions.

SHAPE TRANSFORMATIONS:
    Input:  (B, 3, 224, 224)     — batch of RGB images
    ↓ Reshape into patches
    Step 1: (B, 3, 14, 16, 14, 16) — split H and W into (num_patches, patch_size)
    Step 2: (B, 14, 14, 3, 16, 16) — rearrange so spatial patches are first
    Step 3: (B, 196, 768)           — flatten patches and merge spatial dims
    ↓ Linear projection
    Output: (B, 196, D)             — sequence of patch embeddings

COMPLEXITY ANALYSIS:
    Let N = (H/P)² = number of patches, F = C·P² = flattened patch dim, D = embed_dim

    Reshape/Flatten: O(B · N · F) — just memory reinterpretation, effectively free
    Linear Projection: O(B · N · F · D)
        - For our defaults: O(B · 196 · 768 · 768) ≈ O(B · 115M) multiply-adds
        - This is dominated by the matrix multiplication

    Memory: O(F · D) for the weight matrix = 768 × 768 = 589,824 parameters

COMMON MISTAKES:
    1. Forgetting that images must be exactly divisible by patch size.
       224 / 16 = 14 ✓   but 225 / 16 = 14.0625 ✗
    2. Getting the reshape order wrong — you must keep channels with their
       spatial positions. The reshape is NOT arbitrary.
    3. Confusing Conv2d-based patching with manual patching:
       - nn.Conv2d(3, D, kernel_size=16, stride=16) achieves the same result
         as flatten + linear, but hides the math. We do it manually here.
    4. Not matching flattened patch dimension to linear layer input size.
================================================================================
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Converts a batch of images into a sequence of patch embeddings.

    This is the first layer of a Vision Transformer. It splits each image
    into non-overlapping patches, flattens them, and projects each patch
    into a D-dimensional embedding space.

    Args:
        image_size: Height (and width) of the input image. Must be square.
        patch_size: Height (and width) of each patch. Must evenly divide image_size.
        in_channels: Number of input channels (3 for RGB).
        embed_dim: Dimension of the output embedding for each patch.

    Example:
        >>> patch_embed = PatchEmbedding(image_size=224, patch_size=16, in_channels=3, embed_dim=768)
        >>> images = torch.randn(2, 3, 224, 224)  # batch of 2 RGB images
        >>> embeddings = patch_embed(images)        # shape: (2, 196, 768)
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()

        # ── Validate that the image can be evenly divided into patches ──
        # If image_size is not divisible by patch_size, we'd get fractional
        # patches, which makes no sense geometrically.
        assert image_size % patch_size == 0, (
            f"Image size ({image_size}) must be divisible by "
            f"patch size ({patch_size})."
        )

        self.image_size: int = image_size
        self.patch_size: int = patch_size
        self.in_channels: int = in_channels
        self.embed_dim: int = embed_dim

        # Number of patches along each spatial dimension
        # For 224 / 16 = 14 patches per side
        self.num_patches_per_side: int = image_size // patch_size

        # Total number of patches = 14 × 14 = 196
        self.num_patches: int = self.num_patches_per_side ** 2

        # Each patch has shape (C, P, P). When flattened: C × P × P values.
        # For RGB 16×16 patches: 3 × 16 × 16 = 768
        self.flattened_patch_dim: int = in_channels * patch_size * patch_size

        # ── Linear Projection Layer ──
        # This is the learnable part. It takes a flattened patch (768-dim vector)
        # and projects it into the embedding space (also 768-dim by default,
        # but these are DIFFERENT 768s — one is pixels, the other is learned features).
        #
        # Why linear and not something fancier?
        # The original ViT paper showed that a simple linear projection works
        # surprisingly well. The transformer layers that come after this are
        # powerful enough to learn useful representations from linearly projected patches.
        self.projection = nn.Linear(
            in_features=self.flattened_patch_dim,
            out_features=self.embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a batch of images into a sequence of patch embeddings.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        batch_size, channels, height, width = x.shape

        # ── Validate input dimensions ──
        assert height == self.image_size and width == self.image_size, (
            f"Expected image size ({self.image_size}, {self.image_size}), "
            f"but got ({height}, {width})."
        )
        assert channels == self.in_channels, (
            f"Expected {self.in_channels} channels, but got {channels}."
        )

        print(f"\n{'='*60}")
        print(f"PATCH EMBEDDING -- Forward Pass")
        print(f"{'='*60}")
        print(f"Input shape: {x.shape}")
        print(f"  => (batch={batch_size}, channels={channels}, "
              f"height={height}, width={width})")

        # =====================================================================
        # STEP 1: Reshape the image into a grid of patches
        # =====================================================================
        # We want to split the H dimension into (num_patches_per_side, patch_size)
        # and the W dimension into (num_patches_per_side, patch_size).
        #
        # (B, C, H, W) → (B, C, n_h, P, n_w, P)
        # where n_h = n_w = num_patches_per_side, P = patch_size
        #
        # This is a VIEW operation — no data is copied, just reinterpreted.
        x = x.reshape(
            batch_size,
            channels,
            self.num_patches_per_side,  # 14 groups along height
            self.patch_size,            # 16 pixels per group (height)
            self.num_patches_per_side,  # 14 groups along width
            self.patch_size,            # 16 pixels per group (width)
        )
        print(f"\nAfter reshape into patch grid: {x.shape}")
        print(f"  => (batch={batch_size}, channels={channels}, "
              f"n_patches_h={self.num_patches_per_side}, patch_h={self.patch_size}, "
              f"n_patches_w={self.num_patches_per_side}, patch_w={self.patch_size})")

        # =====================================================================
        # STEP 2: Rearrange dimensions to group patch spatial info together
        # =====================================================================
        # Current:  (B, C,    n_h, P_h,  n_w, P_w)
        # Indices:   0  1      2    3      4    5
        #
        # We want:  (B, n_h,  n_w, C,    P_h, P_w)
        # Why? We want each patch to contain ALL its channels and ALL its pixels
        # grouped together, so we can flatten each patch into a single vector.
        #
        # Think of it as: for each patch position (n_h, n_w), collect all the
        # pixel data (C, P_h, P_w) that belongs to that patch.
        x = x.permute(0, 2, 4, 1, 3, 5)
        print(f"\nAfter permute (group patch data): {x.shape}")
        print(f"  => (batch={batch_size}, "
              f"n_patches_h={self.num_patches_per_side}, "
              f"n_patches_w={self.num_patches_per_side}, "
              f"channels={channels}, "
              f"patch_h={self.patch_size}, "
              f"patch_w={self.patch_size})")

        # =====================================================================
        # STEP 3: Flatten patches into a sequence of vectors
        # =====================================================================
        # (B, n_h, n_w, C, P_h, P_w) → (B, N, F)
        # where N = n_h × n_w = 196 (total patches)
        #       F = C × P_h × P_w = 768 (flattened patch dimension)
        #
        # .contiguous() is needed because permute() creates a non-contiguous
        # view in memory (it changes strides but doesn't move data).
        # reshape() requires contiguous memory, so we must call this first.
        #
        # Why contiguous matters:
        # After permute, the logical order of elements differs from their
        # physical layout in memory. contiguous() rearranges the physical
        # memory to match the logical order, enabling reshape to work.
        x = x.contiguous().reshape(batch_size, self.num_patches, self.flattened_patch_dim)
        print(f"\nAfter flatten into sequence: {x.shape}")
        print(f"  => (batch={batch_size}, "
              f"num_patches={self.num_patches}, "
              f"flattened_dim={self.flattened_patch_dim})")
        print(f"  Each of the {self.num_patches} patches is now a "
              f"{self.flattened_patch_dim}-dim vector")

        # =====================================================================
        # STEP 4: Linear projection into embedding space
        # =====================================================================
        # (B, N, F) → (B, N, D)
        # where D = embed_dim
        #
        # This is the LEARNABLE part of patch embedding.
        # The linear layer learns to extract meaningful features from raw pixel
        # values. Each flattened patch (raw pixels) is projected into a rich
        # embedding space where the transformer can process it.
        #
        # Mathematically: embedding_i = W @ patch_i + b
        # where W ∈ ℝ^(D × F), b ∈ ℝ^D
        #
        # Why not just use the flattened pixels directly?
        # Raw pixel values are not a good representation for attention.
        # The linear projection learns to map pixels into a space where
        # semantically similar patches have similar embeddings.
        x = self.projection(x)
        print(f"\nAfter linear projection: {x.shape}")
        print(f"  => (batch={batch_size}, "
              f"num_patches={self.num_patches}, "
              f"embed_dim={self.embed_dim})")
        print(f"  Each patch is now a {self.embed_dim}-dim learned embedding")
        print(f"{'='*60}\n")

        return x


# =============================================================================
# STANDALONE TEST — Run this file directly to see the shapes in action
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING PATCH EMBEDDING MODULE")
    print("=" * 60)

    # Configuration matching the original ViT-Base
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    IN_CHANNELS = 3
    EMBED_DIM = 768
    BATCH_SIZE = 2

    # Create the patch embedding layer
    patch_embed = PatchEmbedding(
        image_size=IMAGE_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        embed_dim=EMBED_DIM,
    )

    # Create a dummy batch of images
    # In practice, these would be real images normalized to [0, 1] or [-1, 1]
    dummy_images = torch.randn(BATCH_SIZE, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    print(f"\nCreated {BATCH_SIZE} dummy images of size "
          f"{IN_CHANNELS}x{IMAGE_SIZE}x{IMAGE_SIZE}")

    # Run forward pass
    embeddings = patch_embed(dummy_images)

    # Verify output shape
    expected_num_patches = (IMAGE_SIZE // PATCH_SIZE) ** 2
    assert embeddings.shape == (BATCH_SIZE, expected_num_patches, EMBED_DIM), (
        f"Shape mismatch! Expected {(BATCH_SIZE, expected_num_patches, EMBED_DIM)}, "
        f"got {embeddings.shape}"
    )

    print("[OK] Output shape verified!")
    print(f"\nModule Parameters:")
    total_params = sum(p.numel() for p in patch_embed.parameters())
    print(f"  Linear layer weight: {patch_embed.projection.weight.shape} "
          f"= {patch_embed.projection.weight.numel():,} params")
    print(f"  Linear layer bias:   {patch_embed.projection.bias.shape} "
          f"= {patch_embed.projection.bias.numel():,} params")
    print(f"  Total parameters:    {total_params:,}")
