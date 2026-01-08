# models/m2mvt.py
# M2MVT with episodic memory (Fig. 6 faithful)

import torch
import torch.nn as nn

from models.video_transformer import VideoTransformer
from models.gaze_transformer import GazeTransformer
from models.memory_encoder import MemoryAugmentedEncoder


class M2MVT(nn.Module):
    """
    Multi-Modal Multi-View Transformer with Episodic Memory
    """

    def __init__(
        self,
        embed_dim=768,
        gaze_embed_dim=256,
        depth=4,
        num_heads=12,
        num_classes=7,
        num_tokens=1568,
        gaze_tokens=16,
        num_memory_tokens=4,
        views=None
    ):
        super().__init__()

        if views is None:
            views = [
                "Front_View",
                "Left_View",
                "Right_View",
                "Rear_View",
                "Driver_View"
            ]

        self.views = views

        # -------------------------
        # Per-view video encoders
        # -------------------------
        self.view_models = nn.ModuleDict({
            view: VideoTransformer(
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                num_classes=num_classes,
                num_tokens=num_tokens
            )
            for view in views
        })

        # -------------------------
        # Gaze encoder
        # -------------------------
        self.gaze_model = GazeTransformer(
            embed_dim=gaze_embed_dim,
            depth=2,
            num_heads=4,
            num_tokens=gaze_tokens
        )

        # -------------------------
        # Episodic memory encoders
        # -------------------------
        self.mv_encoder = MemoryAugmentedEncoder(
            embed_dim=embed_dim,
            num_memory_tokens=num_memory_tokens,
            depth=depth,
            num_heads=num_heads
        )

        self.ev_encoder = MemoryAugmentedEncoder(
            embed_dim=gaze_embed_dim,
            num_memory_tokens=num_memory_tokens,
            depth=depth,
            num_heads=4
        )

        # -------------------------
        # Final classifier
        # -------------------------
        fusion_dim = embed_dim + gaze_embed_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, batch):
        """
        batch["views"][view]: (B,T,3,H,W)
        batch["gaze"]: (B,T,2)
        """

        # ===== Multi-view stream =====
        mv_cls_tokens = []

        for view in self.views:
            video = batch["views"][view]
            cls = self.view_models[view](video, return_cls=True)
            mv_cls_tokens.append(cls)

        # Stack CLS tokens as sequence
        mv_tokens = torch.stack(mv_cls_tokens, dim=1)
        # Shape: (B, V, D)

        _, mv_out = self.mv_encoder(mv_tokens)
        mv_cls = mv_out.mean(dim=1)
        # Aggregated multi-view CLS

        # ===== Ego + gaze stream =====
        gaze_cls = self.gaze_model(batch["gaze"])
        gaze_tokens = gaze_cls.unsqueeze(1)

        _, ev_out = self.ev_encoder(gaze_tokens)
        ev_cls = ev_out[:, 0]

        # ===== Final prediction =====
        fused = torch.cat([mv_cls, ev_cls], dim=1)
        out = self.classifier(fused)

        return out
