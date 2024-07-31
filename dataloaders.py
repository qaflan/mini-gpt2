import tiktoken
import torch
import os


import numpy as np
import logging
from tqdm import tqdm

logging.getLogger().setLevel(logging.INFO)


def load_np(file: str):
    data = np.load(file)
    data_tensor = torch.tensor(data, dtype=torch.long)
    return data_tensor


class DataLoader:
    def __init__(
        self,
        path: str,
        batch_size: int,
        block_size: int,
        split: str,
        rank: int = 0,
        world_size: int = 1,
        limit_files: int = -1,
    ) -> None:
        assert split in ("train", "validation", "val")

        files = [f for f in os.listdir(path) if split in f]
        logging.info(f"found {len(files)} file(s) for split {split}. Loading...")
        if limit_files > 0:
            logging.info(
                f"will use a max of {limit_files} file(s) since it is explicitly asked."
            )
            files = files[:limit_files]
        self.shards = [
            load_np(os.path.join(path, file_name)) for file_name in tqdm(files)
        ]
        logging.info("done loading")
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.reset()

    def next_batch(self):
        B = self.batch_size
        T = self.block_size
        current_shard = self.shards[self.current_shard_ix]
        buf = current_shard[self.current_pos : B * T + self.current_pos + 1]
        x = buf[:-1].view(-1, T)
        y = buf[1:].view(-1, T)
        self.current_pos += B * T * self.world_size
        if self.current_pos + (B * T * self.world_size + 1) > current_shard.size(0):
            self._reset_pos()
            self.current_shard_ix = (self.current_shard_ix + 1) % len(self.shards)
        return x, y

    def reset(self):
        self._reset_pos()
        self.current_shard_ix = 0

    def _reset_pos(self):
        self.current_pos = self.rank * self.batch_size * self.block_size


class DataLoaderTokenizer:
    def __init__(
        self,
        file_name: str,
        batch_size: int,
        block_size: int,
        model_name: str = "gpt2",
        device: str = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        with open(file_name, "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding(model_name)
        self.data = torch.tensor(tokenizer.encode(text))
        if device:
            self.data = self.data.to(device)
        logging.info(f"Loaded {self.data.size(0)} tokens.")
        self.n_tokens = self.data.size(0)
        self.batch_size = batch_size
        self.block_size = block_size
        self.rank = rank
        self.world_size = world_size
        self.current_pos = self.rank * self.batch_size * self.block_size

    def next_batch(self):
        B = self.batch_size
        T = self.block_size
        buf = self.data[self.current_pos : B * T + self.current_pos + 1]
        x = buf[:-1].view(-1, T)
        y = buf[1:].view(-1, T)
        self.current_pos += B * T * self.world_size
        if self.current_pos + (B * T * self.world_size + 1) > self.data.size(0):
            self.current_pos = self.rank * self.batch_size * self.block_size
        return x, y
