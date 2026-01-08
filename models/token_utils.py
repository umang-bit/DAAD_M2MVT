# models/token_utils.py
# Adds CLS token and positional embeddings to token sequence

import torch
import torch.nn as nn


class AddCLSandPosition(nn.Module):
    """
    Adds a learnable CLS token and positional embeddings.
    """

    def __init__(self, embed_dim, num_tokens):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )
        # One learnable CLS token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_tokens + 1, embed_dim)
        )
        # Positional embedding for CLS + all tokens

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        """
        x shape: (B, N, D)
        """

        B, N, D = x.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)
        # Shape: (B, 1, D)

        x = torch.cat((cls_tokens, x), dim=1)
        # Shape: (B, N+1, D)

        x = x + self.pos_embed[:, : N + 1]
        # Add positional information

        return x
