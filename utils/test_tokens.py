from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.tubelet_embed import TubeletEmbedding
from models.token_utils import AddCLSandPosition

dataset = DAADDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=daad_collate_fn)

batch = next(iter(loader))

video = batch["views"]["Front_View"]

tubelet = TubeletEmbedding()
tokens = tubelet(video)
# (B, 1568, 768)

token_adder = AddCLSandPosition(
    embed_dim=768,
    num_tokens=tokens.shape[1]
)

tokens = token_adder(tokens)

print(tokens.shape)
