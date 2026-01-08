import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.video_transformer import VideoTransformer
from config import MANEUVER_CLASSES


# -------------------------
# 1. Device (CPU or GPU)
# -------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# 2. Dataset & DataLoader
# -------------------------

dataset = DAADDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=daad_collate_fn
)


# -------------------------
# 3. Model
# -------------------------

model = VideoTransformer(
    embed_dim=768,
    depth=4,
    num_heads=12,
    num_classes=len(MANEUVER_CLASSES)
)

model = model.to(device)
# Moves model weights to CPU or GPU


# -------------------------
# 4. Loss function
# -------------------------

criterion = nn.CrossEntropyLoss()
# Used for multi-class classification


# -------------------------
# 5. Optimizer
# -------------------------

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-4
)
# Adam adjusts weights based on gradients


# -------------------------
# 6. One training step
# -------------------------

model.train()
# Puts model in training mode

batch = next(iter(loader))

video = batch["views"]["Front_View"].to(device)
labels = batch["labels"].to(device)

# Forward pass
outputs = model(video)
# Shape: (B, num_classes)

loss = criterion(outputs, labels)
# Compares predictions with true labels

print("Loss:", loss.item())


# -------------------------
# 7. Backpropagation
# -------------------------

optimizer.zero_grad()
# Clears old gradients

loss.backward()
# Computes gradients (how wrong each weight was)

optimizer.step()
# Updates model weights
