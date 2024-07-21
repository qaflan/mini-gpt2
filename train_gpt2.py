# for backward-compatibility with python<3.10
from __future__ import annotations
import tiktoken
import logging
from configs import GPTConfig, GPTTrainConfig
import torch
import time

logging.getLogger().setLevel(logging.INFO)
from gpt import GPT, GPTGenerator


def detect_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    return device


def count_params(model):
    param_count = sum([p.nelement() for p in model.parameters()])
    return param_count


def get_memory_size(model):
    n_bytes = sum([p.nelement() * p.element_size() for p in model.parameters()])
    return n_bytes


class DataLoader:
    def __init__(
        self,
        file_name: str,
        batch_size: int,
        block_size: int,
        model_name: str = "gpt2",
        device: str | None = None,
    ) -> None:
        with open(file_name, "r") as f:
            text = f.read()
        tokenizer = tiktoken.get_encoding(model_name)
        self.data = torch.tensor(tokenizer.encode(text))
        if device:
            self.data = self.data.to(device)
        logging.info(f"Loaded {self.data.size(0)} tokens.")
        self.batch_size = batch_size
        self.block_size = block_size
        self.current_pos = 0

    def next_batch(self):
        B = self.batch_size
        T = self.block_size
        buf = self.data[self.current_pos : B * T + self.current_pos + 1]
        x = buf[:-1].view(-1, T)
        y = buf[1:].view(-1, T)
        self.current_pos += B * T
        if self.current_pos + (B * T + 1) > self.data.size(0):
            self.current_pos = 0
        return x, y


if __name__ == "__main__":
    device = detect_device()
    SEED = 1337
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    logging.info(f"Using {device=}")

    torch.set_float32_matmul_precision("high")

    tokenizer = tiktoken.get_encoding("gpt2")
    # gpt = GPT.from_pretrained("gpt2")
    gpt = GPT(GPTConfig())
    logging.info(f"{gpt.config=}")
    block_size = gpt.config.block_size
    train_config = GPTTrainConfig()
    n_iterations = train_config.n_iterations
    batch_size = train_config.batch_size
    n_steps = 20  # int(n_iterations * 330 // batch_size)
    logging.info(f"found {count_params(gpt)} parameters")
    logging.info(f"parameters size ~ {get_memory_size(gpt) / 1024 / 1024:.2f} MB")
    gpt.to(device)

    data_loader = DataLoader(
        file_name="input.txt",
        model_name="gpt2",
        batch_size=batch_size,
        block_size=block_size,
    )
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=3e-4)
    for i in range(n_steps):
        x, y = data_loader.next_batch()
        time0 = time.time()
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = gpt(x, y)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        time1 = time.time()
        total_time = time1 - time0
        throughput = (train_config.batch_size * gpt.config.block_size) / total_time
        logging.info(
            f"step {i:3d}, loss: {loss.item():4f} took {total_time*1000:.2f} milliseconds @ {throughput:.2f} tokens/sec"
        )

    seed_text = "Hello, I am a language model,"
    gpt.eval()
    torch.manual_seed(42)
    my_generator = GPTGenerator(gpt, tokenizer, device)
    for sentence in my_generator.generate(seed_text, 100, 3):
        print(sentence)
        print("_" * 40)
