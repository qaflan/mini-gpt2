# for backward-compatibility with python<3.10
from __future__ import annotations
import tiktoken
import logging
from configs import GPTConfig, GPTTrainConfig
import torch

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
    return sum([torch.prod(torch.tensor(p.size())) for p in model.parameters()])


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
    # device = "cpu" # override
    logging.info(f"Using {device=}")

    tokenizer = tiktoken.get_encoding("gpt2")
    gpt = GPT.from_pretrained("gpt2")
    # gpt = GPT(GPTConfig(block_size=1024))
    logging.info(f"{gpt.config=}")
    block_size = gpt.config.block_size
    train_config = GPTTrainConfig()
    n_iterations = train_config.n_iterations
    batch_size = train_config.batch_size
    n_steps = int(n_iterations * 330 // batch_size)
    logging.info(f"found {count_params(gpt)} parameters")
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
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        logits, loss = gpt(x, y)
        loss.backward()
        optimizer.step()
        logging.info(f"step {i}, loss: {loss.item()}")

    seed_text = "Hello, I am a language model,"
    gpt.eval()
    torch.manual_seed(42)
    my_generator = GPTGenerator(gpt, tokenizer, device)
    for sentence in my_generator.generate(seed_text, 100, 3):
        print(sentence)
        print("_" * 40)
