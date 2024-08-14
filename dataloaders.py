import tiktoken
import torch
import datasets
import os
from utils import log, IS_MASTER

import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from typing import Union


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
        log(f"found {len(files)} file(s) for split {split}. Loading...")
        if limit_files > 0:
            log(
                f"will use a max of {limit_files} file(s) since it is explicitly asked."
            )
            files = files[:limit_files]
        if IS_MASTER:
            files = tqdm(files)
        self.shards = [load_np(os.path.join(path, file_name)) for file_name in files]
        log("done loading")
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
        device: Union[str, None] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        with open(file_name, "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding(model_name)
        self.data = torch.tensor(tokenizer.encode(text))
        if device:
            self.data = self.data.to(device)
        log(f"Loaded {self.data.size(0)} tokens.")
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


class HellaSwagLoader:
    def __init__(
        self,
        tokenizer: tiktoken.core.Encoding,
        path: str = "Rowan/hellaswag",
        split: str = "train",
        rank: int = 0,
        world_size: int = 1,
    ):
        self.ds = datasets.load_dataset(path=path, split=split)
        self.tokenizer = tokenizer
        self.ds_prepared = self.preapre_records()
        self.rank = rank
        self.world_size = world_size
        self.pos = self.rank
        self.n = len(self.ds_prepared) // self.world_size
        self.reset()

    def preapre_records(self):
        ds_prepared = []
        ds = self.ds
        if IS_MASTER:
            ds = tqdm(self.ds, desc="Preparing HellaSwag records")
        i = 0
        for item in ds:
            ds_prepared.append(self._prepare(item))
            i += 1
            if i > 20:
                break
        return ds_prepared

    def reset(self):
        self.pos = self.rank

    def _prepare(self, q):
        padding_value = 0
        ctx_encoded = torch.tensor(self.tokenizer.encode_ordinary(q["ctx"]))
        endings_encoded = [
            torch.tensor(self.tokenizer.encode_ordinary(" " + ending))
            for ending in q["endings"]
        ]

        full_encoded = [torch.cat([ctx_encoded, ending]) for ending in endings_encoded]
        max_length = max(sentence.size(0) for sentence in full_encoded)
        full_encoded = torch.stack(
            [
                F.pad(
                    sentence,
                    pad=(0, max_length - sentence.size(0)),
                    value=padding_value,
                )
                for sentence in full_encoded
            ]
        )

        ctx_mask = torch.zeros_like(ctx_encoded)
        encodings_mask = [
            torch.cat([ctx_mask, torch.ones_like(ending)]) for ending in endings_encoded
        ]
        full_mask = torch.stack(
            [F.pad(m, pad=(0, max_length - m.size(0)), value=0) for m in encodings_mask]
        )

        return {"tokens": full_encoded, "mask": full_mask, "label": int(q["label"])}

    def next_batch(self):
        result = self.ds_prepared[self.pos]
        self.pos = (self.pos + self.world_size) % len(self.ds_prepared)
        return result
