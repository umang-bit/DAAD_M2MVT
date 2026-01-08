# config.py
# This file stores all configuration values in one place

from pathlib import Path  # Helps handle file paths safely across OS

# -----------------------
# Project paths
# -----------------------

PROJECT_ROOT = Path(__file__).parent
# Path to the root of this project (DAAD_M2MVT)

DATA_ROOT = PROJECT_ROOT / "data" / "daad"
# Path where the raw DAAD dataset lives

# -----------------------
# Views (do NOT assume availability)
# -----------------------

VIEWS = [
    "Front_View",
    "Left_View",
    "Right_View",
    "Rear_View",
    "Driver_View",
]
# These are the camera views described in the paper

GAZE_VIEW = "Gaze_View"
# Gaze is a separate modality (Project Aria)

# -----------------------
# Maneuver classes (full set, no bias)
# -----------------------

MANEUVER_CLASSES = [
    "go_straight",
    "left_turn",
    "right_turn",
    "left_change",
    "right_change",
    "slow_stop",
    "u_turn",
]
# Even if we only have left_change now, we define all classes

# -----------------------
# Video sampling parameters
# -----------------------

FPS = 30                 # DAAD videos are recorded at 30 FPS
NUM_FRAMES = 16          # Number of frames per clip (paper uses 16×4, 32×3)
FRAME_STRIDE = 4         # Temporal stride between frames
IMAGE_SIZE = 224         # Input resolution for transformers

# -----------------------
# Training parameters
# -----------------------

BATCH_SIZE = 2           # Small batch for now (safe for debugging)
DEVICE = "cuda"          # Will fall back to CPU if CUDA unavailable
