# utils/collate.py
# Defines how multiple DAAD samples are combined into a batch

import torch


def daad_collate_fn(batch):
    """
    batch: list of samples returned by DAADDataset

    Each sample:
    {
        "views": {view_name: tensor},
        "gaze": tensor,
        "label": int
    }
    """

    batch_views = {}
    batch_gaze = []
    batch_labels = []

    # Initialize view lists
    for view in batch[0]["views"]:
        batch_views[view] = []

    # Collect data
    for sample in batch:
        for view, video in sample["views"].items():
            batch_views[view].append(video)

        batch_gaze.append(sample["gaze"])
        batch_labels.append(sample["label"])

    # Stack views
    for view in batch_views:
        batch_views[view] = torch.stack(batch_views[view])
        # Shape: (B, T, 3, 224, 224)

    batch_gaze = torch.stack(batch_gaze)
    # Shape: (B, T, 2)

    batch_labels = torch.tensor(batch_labels)
    # Shape: (B,)

    return {
        "views": batch_views,
        "gaze": batch_gaze,
        "labels": batch_labels
    }
