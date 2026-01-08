from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.m2mvt import M2MVT
from config import MANEUVER_CLASSES

dataset = DAADDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=daad_collate_fn
)

batch = next(iter(loader))

model = M2MVT(
    embed_dim=768,
    depth=2,  # keep small for test
    num_heads=8,
    num_classes=len(MANEUVER_CLASSES)
)

output = model(batch)

print("Output shape:", output.shape)
