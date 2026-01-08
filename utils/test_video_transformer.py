from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.video_transformer import VideoTransformer
from config import MANEUVER_CLASSES

dataset = DAADDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=daad_collate_fn
)

batch = next(iter(loader))

video = batch["views"]["Front_View"]
labels = batch["labels"]

model = VideoTransformer(
    embed_dim=768,
    depth=4,
    num_heads=12,
    num_classes=len(MANEUVER_CLASSES)
)

output = model(video)

print("Output shape:", output.shape)
print("Labels shape:", labels.shape)
