# utils/gaze_utils.py
# Robust gaze loader for DAAD EyeGaze ZIPs

import zipfile
import pandas as pd
import torch
import os


def load_gaze_from_zip(zip_path, max_len=16):
    """
    Load gaze data from <clip_id>_EyeGaze.zip.
    If gaze CSV is missing, returns zeros.

    Returns:
        torch.Tensor of shape (T, 2)
    """

    # Default fallback (no gaze)
    fallback = torch.zeros((max_len, 2), dtype=torch.float32)

    if not os.path.exists(zip_path):
        return fallback

    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            # List all files inside ZIP
            names = z.namelist()

            # Try all known gaze CSV variants
            csv_name = None
            for name in names:
                lname = name.lower()
                if (
                    "generalized_eye_gaze.csv" in lname
                    or "eyegaze.csv" in lname
                ):
                    csv_name = name
                    break

            # If no gaze CSV found → fallback
            if csv_name is None:
                return fallback

            # Read CSV directly from ZIP
            with z.open(csv_name) as f:
                df = pd.read_csv(f)

        # Ensure required columns exist
        required_cols = ["yaw_rads_cpf", "pitch_rads_cpf"]
        for col in required_cols:
            if col not in df.columns:
                return fallback

        gaze = df[required_cols].values  # (N, 2)

        # Truncate or pad
        if gaze.shape[0] >= max_len:
            gaze = gaze[:max_len]
            return torch.tensor(gaze, dtype=torch.float32)

        pad_len = max_len - gaze.shape[0]
        pad = torch.zeros((pad_len, 2), dtype=torch.float32)
        gaze = torch.cat(
            [torch.tensor(gaze, dtype=torch.float32), pad],
            dim=0
        )
        return gaze

    except Exception:
        # Any parsing / zip / csv error → fallback
        return fallback
