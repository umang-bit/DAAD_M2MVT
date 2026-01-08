# datasets/daad_dataset.py
# This file defines how DAAD data is loaded into PyTorch

import os
import torch
from torch.utils.data import Dataset

from config import DATA_ROOT, VIEWS, MANEUVER_CLASSES
from utils.video_utils import load_video
from utils.gaze_utils import load_gaze_from_zip


class DAADDataset(Dataset):
    """
    PyTorch Dataset for DAAD.
    One dataset item = one clip ID across all views + gaze.
    """

    def __init__(self):
        """
        This runs ONCE when the dataset object is created.
        Its job is to discover what data exists.
        """

        self.samples = []
        # Stores (clip_id, class_name)

        # --------------------------------------------------
        # STEP 1: Find clip IDs using Front_View as reference
        # --------------------------------------------------

        reference_view = "Front_View"
        reference_view_path = DATA_ROOT / reference_view

        for class_name in os.listdir(reference_view_path):
            class_path = reference_view_path / class_name

            if not class_path.is_dir():
                continue

            for filename in os.listdir(class_path):
                if not filename.endswith(".mp4"):
                    continue

                clip_id = filename.replace(".mp4", "")
                self.samples.append((clip_id, class_name))

        # --------------------------------------------------
        # STEP 2: Map class names to numeric labels
        # --------------------------------------------------

        self.class_to_idx = {}
        for idx, name in enumerate(MANEUVER_CLASSES):
            self.class_to_idx[name.lower()] = idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Returns ONE sample.
        """

        clip_id, class_name = self.samples[index]

        label = self.class_to_idx[class_name.lower()]

        views_data = {}

        # --------------------------------------------------
        # STEP 3: Load all camera views
        # --------------------------------------------------

        for view in VIEWS:
            video_path = (
                DATA_ROOT
                / view
                / class_name
                / f"{clip_id}.mp4"
            )

            if video_path.exists():
                video_tensor = load_video(str(video_path))
                # Shape: (T, 3, 224, 224)
            else:
                video_tensor = torch.zeros((16, 3, 224, 224))

            views_data[view] = video_tensor

        # --------------------------------------------------
        # STEP 4: Load REAL gaze data from EyeGaze zip
        # --------------------------------------------------

        gaze_zip_path = (
            DATA_ROOT
            / "Gaze_View"
            / class_name
            / f"{clip_id}_EyeGaze.zip"
        )

        if gaze_zip_path.exists():
            gaze_tensor = load_gaze_from_zip(
                str(gaze_zip_path),
                max_len=16
            )
            # Shape: (16, 2)
        else:
            gaze_tensor = torch.zeros((16, 2))

        return {
            "views": views_data,
            "gaze": gaze_tensor,
            "label": label,
            "clip_id": clip_id,
        }
