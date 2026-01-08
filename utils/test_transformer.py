from torch.utils.data import DataLoader
from datasets.daad_dataset import DAADDataset
from utils.collate import daad_collate_fn
from models.tubelet_embed import TubeletEmbedding
from models.token_utils import AddCLSandPosition
from models.transformer_block import TransformerEncoderBlock

dataset = DAADDataset()
loader = DataLoader(dataset, batch_size=2, collate_fn=daad_collate_fn)

batch = next(iter(loader))

video = batch["views"]["Front_View"]

tubelet = TubeletEmbedding()
tokens = tubelet(video)

adder = AddCLSandPosition(
    embed_dim=768,
    num_tokens=tokens.shape[1]
)
tokens = adder(tokens)

block = TransformerEncoderBlock(
    embed_dim=768,
    num_heads=12
)

out = block(tokens)

print(out.shape)
