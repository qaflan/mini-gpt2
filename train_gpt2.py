# for backward-compatibility with python<3.10
from __future__ import annotations

from gpt import GPT

import tiktoken
import logging
from configs import GPTConfig, GPTTrainConfig, OptimizerConfig, Config
import torch
import time
import wandb
import math
import inspect


logging.getLogger().setLevel(logging.INFO)


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
        self.n_tokens = self.data.size(0)
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


def get_lr_for_step(optimizer_config: OptimizerConfig, step: int) -> float:
    # step: int, warmup_steps: int, min_lr: float, max_lr: float, max_steps: int
    warmup_steps = optimizer_config.warmup_steps
    max_lr = optimizer_config.max_lr
    min_lr = optimizer_config.min_lr
    max_steps = optimizer_config.max_steps
    if step < warmup_steps:
        return (
            max_lr * (step + 1) / warmup_steps
        )  # linearly increases the lr so that at `warmup_steps` the value is `max_lr``
    if step > max_steps:
        return min_lr

    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def get_optimizer(
    optimizer_config: OptimizerConfig, model: torch.nn.Module, device: str
):
    param_dict = {
        param_name: p for param_name, p in model.named_parameters() if p.requires_grad
    }
    weight_decay = optimizer_config.weight_decay
    decay_params = [p for p in param_dict.values() if p.dim() >= 2]
    nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    logging.info(
        f"Using weight decay of {weight_decay} for {len(decay_params)} tensors ({sum([p.numel() for p in decay_params])} parameters)"
    )
    logging.info(
        f"Using no weight decay for {len(nodecay_params)} tensors ({sum([p.numel() for p in nodecay_params])} parameters)"
    )
    use_fused = (
        "cuda" in device and "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    if use_fused:
        logging.info("Using fused version of the optimizer")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=1e-8,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        fused=use_fused,
    )
    return optimizer


if __name__ == "__main__":
    USE_WANDB = True
    train_config = GPTTrainConfig()
    optimizer_config = OptimizerConfig()
    model_config = GPTConfig(vocab_size=50304)
    config = Config(
        model_config=model_config,
        train_config=train_config,
        optimizer_config=optimizer_config,
    )
    if USE_WANDB:
        wandb.init(project="gpt2", name="debug GPT Train 10 epochs", config=config)
    device = detect_device()
    if train_config.seed is not None:
        torch.manual_seed(train_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(train_config.seed)

    logging.info(f"Using {device=}")

    torch.set_float32_matmul_precision(train_config.float32_matmul_precision)

    tokenizer = tiktoken.get_encoding("gpt2")
    # gpt = GPT.from_pretrained("gpt2")
    gpt = GPT(model_config)
    logging.info("compiling torch model...")
    gpt = torch.compile(gpt)
    logging.info(f"{gpt.config=}")

    micro_batch_size = train_config.micro_batch_size
    assert (
        train_config.tokens_per_batch
        % (model_config.block_size * train_config.micro_batch_size)
        == 0
    )
    gradient_accum_batch_size = train_config.tokens_per_batch // (
        model_config.block_size * train_config.micro_batch_size
    )
    logging.info(f"found {count_params(gpt)} parameters")
    logging.info(f"parameters size ~ {get_memory_size(gpt) / 1024 / 1024:.2f} MB")
    gpt.to(device)

    data_loader = DataLoader(
        file_name="input.txt",
        model_name="gpt2",
        batch_size=micro_batch_size,
        block_size=gpt.config.block_size,
    )
    logging.info(f"Will train for {n_steps} steps")
    optimizer = get_optimizer(optimizer_config, gpt, device=device)
    for step in range(n_steps):
        time0 = time.time()
        optimizer.zero_grad()
        total_loss = 0.0
        for _ in range(gradient_accum_batch_size):
            x, y = data_loader.next_batch()
            x = x.to(device)
            y = y.to(device)
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, loss = gpt(x, y)
            loss /= gradient_accum_batch_size
            total_loss += loss.detach()
            loss.backward()

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gpt.parameters(), max_norm=optimizer_config.clip_grad_max_norm
        )
        lr = get_lr_for_step(optimizer_config, step=step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        optimizer.step()
        torch.cuda.synchronize()
        time1 = time.time()
        total_time = time1 - time0
        throughput = train_config.tokens_per_batch / total_time
        log_payload = dict(
            lr=lr, loss=total_loss, throughput=throughput, gradient_norm=gradient_norm
        )
        logging.info(
            f"step {step:3d} took {total_time*1000:.2f} millis | "
            + " | ".join(
                [
                    f"{k}={v:.4e}" if k == "lr" else f"{k}={v:.2f}"
                    for k, v in log_payload.items()
                ]
            )
        )
        if USE_WANDB:
            wandb.log(log_payload)

    # seed_text = "Hello, I am a language model,"
    # gpt.eval()
    # torch.manual_seed(42)
    # my_generator = GPTGenerator(gpt, tokenizer, device)
    # for sentence in my_generator.generate(seed_text, 100, 3):
    #     print(sentence)
    #     print("_" * 40)
