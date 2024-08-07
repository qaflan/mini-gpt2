import os
import logging

import torch

IS_DDP_RUN = "RANK" in os.environ
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
IS_MASTER = RANK == 0


def log(*args, **kwargs):
    if IS_MASTER:
        logging.info(*args, **kwargs)


def detect_device():
    if IS_DDP_RUN:
        device = f"cuda:{RANK}"
        torch.cuda.set_device(device)
        return device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device
