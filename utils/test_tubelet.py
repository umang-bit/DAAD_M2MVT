from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.tubelet_embed import TubeletEmbedding

dataset = DAADDataset()
loader = DataLoader(
    dataset,
    batch_size=2,
    collate_fn=daad_collate_fn
)

batch = next(iter(loader))

video = batch["views"]["Front_View"]
# Shape: (B, T, 3, 224, 224)

tubelet = TubeletEmbedding()
tokens = tubelet(video)

print(tokens.shape)
