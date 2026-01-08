# models/tubelet_embed.py
# Converts a video into a sequence of tokens (tubelet embeddings)

import torch
import torch.nn as nn


class TubeletEmbedding(nn.Module):
    """
    Turns (B, T, C, H, W) video into (B, N, D) tokens.
    """

    def __init__(
        self,
        in_channels=3,
        embed_dim=768,
        tubelet_size=(2, 16, 16),
    ):
        super().__init__()

        self.tubelet_size = tubelet_size
        # tubelet_size = (time, height, width)

        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=tubelet_size,
            stride=tubelet_size
        )
        # Conv3D does:
        # - split video into tubelets
        # - flatten each tubelet
        # - project it to embed_dim

    def forward(self, x):
        """
        x shape: (B, T, C, H, W)
        """

        # Conv3D expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        x = self.proj(x)
        # Shape now: (B, D, T', H', W')
        # where T' = T / tubelet_t, etc.

        x = x.flatten(2)
        # Shape: (B, D, N)
        # N = number of tubelets

        x = x.transpose(1, 2)
        # Shape: (B, N, D)

        return x
