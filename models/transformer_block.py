# models/transformer_block.py
# A single Transformer Encoder block

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """
    One transformer encoder block.
    Input and output shape: (B, N, D)
    """

    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0):
        super().__init__()

        # -------------------------
        # Layer Normalization
        # -------------------------

        self.norm1 = nn.LayerNorm(embed_dim)
        # Normalizes token features before attention

        self.norm2 = nn.LayerNorm(embed_dim)
        # Normalizes token features before MLP

        # -------------------------
        # Multi-Head Self Attention
        # -------------------------

        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        # batch_first=True means input is (B, N, D)

        # -------------------------
        # Feed-Forward Network (MLP)
        # -------------------------

        hidden_dim = int(embed_dim * mlp_ratio)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        # Expands → non-linearity → compresses back

    def forward(self, x):
        """
        x shape: (B, N, D)
        """

        # -------------------------
        # Self-Attention
        # -------------------------

        x_norm = self.norm1(x)
        # Normalize before attention

        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm
        )
        # Each token attends to every other token

        x = x + attn_out
        # Residual connection (skip connection)

        # -------------------------
        # Feed-Forward Network
        # -------------------------

        x_norm = self.norm2(x)
        # Normalize before MLP

        mlp_out = self.mlp(x_norm)
        # Token-wise transformation

        x = x + mlp_out
        # Residual connection

        return x
