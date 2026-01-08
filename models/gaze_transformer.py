# models/gaze_transformer.py
# Transformer for gaze time-series

import torch
import torch.nn as nn

from models.token_utils import AddCLSandPosition
from models.transformer_block import TransformerEncoderBlock


class GazeTransformer(nn.Module):
    """
    Transformer encoder for gaze sequences.
    Input: (B, T, 2)
    Output: CLS token (B, D)
    """

    def __init__(
        self,
        embed_dim=256,
        depth=2,
        num_heads=4,
        num_tokens=16
    ):
        super().__init__()

        # -------------------------
        # Gaze embedding
        # -------------------------
        self.embed = nn.Linear(2, embed_dim)
        # Maps (x, y) → D-dim vector

        # -------------------------
        # CLS + positional embeddings
        # -------------------------
        self.add_tokens = AddCLSandPosition(
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )

        # -------------------------
        # Transformer blocks
        # -------------------------
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, gaze):
        """
        gaze shape: (B, T, 2)
        """

        # Embed gaze points
        x = self.embed(gaze)
        # (B, T, D)

        # Add CLS + positional encoding
        x = self.add_tokens(x)
        # (B, T+1, D)

        # Transformer encoder
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Return CLS token
        cls_gaze = x[:, 0]
        # (B, D)

        return cls_gaze
