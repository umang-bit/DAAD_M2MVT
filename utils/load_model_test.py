import torch
from models.video_transformer import VideoTransformer
from config import MANEUVER_CLASSES

device = torch.device("cpu")

model = VideoTransformer(
    embed_dim=768,
    depth=4,
    num_heads=12,
    num_classes=len(MANEUVER_CLASSES),
    num_tokens=1568
)

checkpoint = torch.load(
    "video_transformer_single_view.pth",
    map_location=device
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

print("Model loaded successfully.")
