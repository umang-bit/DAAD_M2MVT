# models/memory_encoder.py
# Transformer encoder with episodic memory tokens

import torch
import torch.nn as nn
from models.transformer_block import TransformerEncoderBlock


class MemoryAugmentedEncoder(nn.Module):
    """
    Transformer encoder that prepends learnable memory tokens.
    """

    def __init__(
        self,
        embed_dim=768,
        num_memory_tokens=4,
        depth=4,
        num_heads=12
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_memory_tokens = num_memory_tokens

        # Learnable memory tokens (shared across batches)
        self.memory_tokens = nn.Parameter(
            torch.randn(1, num_memory_tokens, embed_dim)
        )

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        tokens: (B, N, D) — includes CLS + modality tokens
        """

        B = tokens.size(0)

        # Expand memory tokens across batch
        memory = self.memory_tokens.expand(B, -1, -1)
        # Shape: (B, K, D)

        # Concatenate memory + tokens
        x = torch.cat([memory, tokens], dim=1)
        # Shape: (B, K+N, D)

        # Transformer encoding
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Split memory and tokens back
        updated_memory = x[:, :self.num_memory_tokens]
        updated_tokens = x[:, self.num_memory_tokens:]

        return updated_memory, updated_tokens
