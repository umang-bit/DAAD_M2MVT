from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.dummy_model import DummyModel
from config import MANEUVER_CLASSES

dataset = DAADDataset()

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=daad_collate_fn
)

batch = next(iter(loader))

model = DummyModel(num_classes=len(MANEUVER_CLASSES))

output = model(batch)

print(output.shape)
