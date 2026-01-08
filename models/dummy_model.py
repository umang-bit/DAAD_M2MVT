# models/dummy_model.py
# A very simple model to understand data flow

import torch
import torch.nn as nn
# nn contains neural network layers


class DummyModel(nn.Module):
    """
    This is NOT a transformer.
    This is only to understand shapes and batching.
    """

    def __init__(self, num_classes):
        super().__init__()
        # Initializes nn.Module internals

        self.classifier = nn.Linear(3, num_classes)
        # This layer maps a 3D vector → class scores
        # Why 3? Because we will average over RGB channels

    def forward(self, batch):
        """
        batch is exactly what DataLoader returns:

        batch["views"]["Front_View"] → (B, T, 3, 224, 224)
        """

        x = batch["views"]["Front_View"]
        # x shape: (B, T, 3, 224, 224)

        # --------------------------------------------------
        # Reduce spatial dimensions
        # --------------------------------------------------

        x = x.mean(dim=-1).mean(dim=-1)
        # Averages over H and W
        # Shape now: (B, T, 3)

        # --------------------------------------------------
        # Reduce temporal dimension
        # --------------------------------------------------

        x = x.mean(dim=1)
        # Averages over time
        # Shape now: (B, 3)

        # --------------------------------------------------
        # Classifier
        # --------------------------------------------------

        out = self.classifier(x)
        # Shape: (B, num_classes)

        return out
