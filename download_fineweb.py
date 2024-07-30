import datasets
import numpy as np
import os
import multiprocessing as mp
from tqdm import tqdm
import tiktoken
import sys
import logging
from configs import GPTDataConfig

logging.getLogger().setLevel(logging.INFO)


def save_shard(output_path: str, shard_idx: int, all_tokens: np.ndarray) -> None:
    name = "validation" if shard_idx == 0 else "train"
    np.save(os.path.join(output_path, f"{name}_{shard_idx:05}"), all_tokens)


enc = tiktoken.get_encoding("gpt2")


def tokenize_document(doc):
    eot = enc._special_tokens["<|endoftext|>"]
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens).astype(np.uint16)
    return tokens_np


def main():
    data_config = GPTDataConfig()
    output_path = data_config.path
    os.makedirs(output_path, exist_ok=True)
    fw = datasets.load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"
    )

    shard_size = int(1e8)
    total_tokens_written = 0
    shard_idx = 0
    pb = tqdm(f"Shard {shard_idx}", total=shard_size)
    with mp.Pool(os.cpu_count() // 2) as pool:
        all_tokens = np.empty(shard_size, dtype=np.uint16)
        idx = 0

        for tokens in pool.imap_unordered(tokenize_document, fw, chunksize=16):
            n_processed = tokens.shape[0]
            if idx + n_processed >= shard_size:  # does not fit
                left_space = all_tokens.shape[0] - idx
                all_tokens[idx : idx + left_space] = tokens[:left_space]
                idx += left_space
                pb.update(left_space)
                sys.stderr.flush()
                save_shard(output_path, shard_idx, all_tokens[:idx])
                shard_idx += 1
                pb = tqdm(f"Shard {shard_idx}", total=shard_size)
                total_tokens_written += idx
                # start a new shard
                idx = 0
                tokens = tokens[left_space:]
                n_processed = tokens.shape[0]
            pb.update(n_processed)
            all_tokens[idx : idx + n_processed] = tokens
            idx += n_processed
        if n_processed > 0:
            logging.info(f"Writing remaining shard of size {idx}")
            save_shard(output_path, shard_idx, all_tokens[:idx])
            total_tokens_written += idx
    logging.info(f"processed {total_tokens_written} tokens")


if __name__ == "__main__":
    main()
