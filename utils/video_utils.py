# utils/video_utils.py
# This file contains helper functions to read and process video files.
# It is NOT meant to be run directly.

import cv2
# cv2 is OpenCV, a library that lets us open and read video files (.mp4)

import torch
# torch is PyTorch, used to create tensors for deep learning models

import numpy as np
# numpy handles numerical arrays (OpenCV frames are NumPy arrays)

from torchvision import transforms
# torchvision.transforms provides common image preprocessing tools

from config import IMAGE_SIZE
# Import IMAGE_SIZE (224) from config so input size is consistent everywhere


# --------------------------------------------------
# Define how each video frame should be processed
# --------------------------------------------------

video_transform = transforms.Compose([
    transforms.ToPILImage(),
    # Converts a NumPy image array into a PIL Image
    # Many torchvision transforms require PIL format

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    # Resizes image to 224x224 (required by transformer models)

    transforms.ToTensor(),
    # Converts image to a PyTorch tensor
    # Also changes shape from (H, W, 3) -> (3, H, W)
    # Also normalizes pixel values from [0,255] to [0,1]
])


# --------------------------------------------------
# Function to load a video and sample frames from it
# --------------------------------------------------

def load_video(video_path, num_frames=16, frame_stride=4):
    """
    Reads a video file and returns a fixed number of frames.

    video_path   : path to the .mp4 video file
    num_frames   : how many frames we want in total
    frame_stride : take every nth frame (temporal sampling)
    """

    cap = cv2.VideoCapture(video_path)
    # Opens the video file so we can read frames one by one

    frames = []
    # This list will store the processed frames we keep

    frame_count = 0
    # Counts how many frames we have read from the video

    while True:
        ret, frame = cap.read()
        # ret  = True if a frame was successfully read
        # frame = the actual image data (NumPy array)

        if not ret:
            break
            # If no frame is returned, the video has ended

        if frame_count % frame_stride == 0:
            # Only keep every `frame_stride`-th frame
            # Example: if stride=4 → keep frames 0,4,8,12,...

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # OpenCV reads images in BGR format
            # Deep learning models expect RGB, so we convert

            frame = video_transform(frame)
            # Resize frame and convert it to a PyTorch tensor

            frames.append(frame)
            # Store the processed frame

            if len(frames) == num_frames:
                break
                # Stop once we have enough frames

        frame_count += 1
        # Move to the next frame index

    cap.release()
    # Close the video file to free system resources

    if len(frames) < num_frames:
        # If the video is too short and we got fewer frames than needed

        last_frame = frames[-1]
        # Take the last available frame

        while len(frames) < num_frames:
            frames.append(last_frame)
            # Repeat the last frame until we reach num_frames
            # This ensures fixed-size input for the model

    video_tensor = torch.stack(frames)
    # Convert list of tensors into a single tensor
    # Final shape: (T, 3, 224, 224)

    return video_tensor
    # Return the video as a PyTorch tensor
