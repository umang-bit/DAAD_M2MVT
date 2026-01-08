from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn

dataset = DAADDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=daad_collate_fn
)

batch = next(iter(loader))

print(batch["views"]["Front_View"].shape)
print(batch["gaze"].shape)
print(batch["labels"])
