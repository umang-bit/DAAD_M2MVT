import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.video_transformer import VideoTransformer
from config import MANEUVER_CLASSES


# -------------------------
# Configuration
# -------------------------
EPOCHS = 10
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_TOKENS = 1568   # must match tubelet setup


# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# -------------------------
# Dataset & Split
# -------------------------
dataset = DAADDataset()

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset, [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=daad_collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=daad_collate_fn
)


# -------------------------
# Model
# -------------------------
model = VideoTransformer(
    embed_dim=768,
    depth=4,
    num_heads=12,
    num_classes=len(MANEUVER_CLASSES),
    num_tokens=NUM_TOKENS
).to(device)


# -------------------------
# Loss & Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# -------------------------
# Training Loop
# -------------------------
model.train()

for epoch in range(EPOCHS):
    total_loss = 0.0

    for batch in train_loader:
        video = batch["views"]["Front_View"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(video)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")


# -------------------------
# Validation
# -------------------------
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for batch in val_loader:
        video = batch["views"]["Front_View"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(video)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")


# -------------------------
# Save model
# -------------------------
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
}

torch.save(checkpoint, "video_transformer_single_view.pth")
print("Model saved.")
