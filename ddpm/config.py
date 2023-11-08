import os
from dataclasses import dataclass
from .helper import get_default_device


@dataclass
class BaseConfig:
    DEVICE = get_default_device()
    DATASET = 'data'

    # For logging inference images and saving checkpoints.
    root_log_dir = os.path.join("logs/", "inference")
    root_checkpoint_dir = os.path.join("logs/", "checkpoints")

    # For sampling.
    root_samples_dir = "samples/"

    # Current log and checkpoint directory.
    log_dir = "version_0"
    checkpoint_dir = "version_0"


@dataclass
class TrainingConfig:
    TIMESTEPS = 1000  # define number of diffusion timesteps
    IMG_SHAPE = (3, 64, 64)
    NUM_EPOCHS = 100
    BATCH_SIZE = 128
    LR = 2e-4
    NUM_WORKERS = 2
