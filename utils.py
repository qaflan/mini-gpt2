import os
import logging

IS_DDP_RUN = "RANK" in os.environ
RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", -1))
IS_MASTER = RANK == 0


def log(*args, **kwargs):
    if IS_MASTER:
        logging.info(*args, **kwargs)
