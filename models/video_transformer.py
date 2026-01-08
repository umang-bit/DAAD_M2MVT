# models/video_transformer.py
# Single-view Video Transformer (clean save/load version)

import torch
import torch.nn as nn

from models.tubelet_embed import TubeletEmbedding
from models.token_utils import AddCLSandPosition
from models.transformer_block import TransformerEncoderBlock


class VideoTransformer(nn.Module):
    """
    End-to-end video transformer for ONE camera view.
    """

    def __init__(
        self,
        embed_dim=768,
        depth=4,
        num_heads=12,
        num_classes=7,
        num_tokens=1568,   # IMPORTANT: fixed number of tubelets
    ):
        super().__init__()

        # -------------------------
        # Tubelet embedding
        # -------------------------
        self.tubelet = TubeletEmbedding(
            in_channels=3,
            embed_dim=embed_dim,
            tubelet_size=(2, 16, 16),
        )

        # -------------------------
        # CLS token + positional embeddings
        # -------------------------
        self.add_tokens = AddCLSandPosition(
            embed_dim=embed_dim,
            num_tokens=num_tokens
        )

        # -------------------------
        # Transformer encoder blocks
        # -------------------------
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # -------------------------
        # Classification head
        # -------------------------
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, video, return_cls=False):
        """
        video shape: (B, T, 3, H, W)
        """

        x = self.tubelet(video)
        x = self.add_tokens(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token = x[:, 0]  # (B, D)

        if return_cls:
            return cls_token

        out = self.head(cls_token)
        return out

