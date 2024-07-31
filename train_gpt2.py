# for backward-compatibility with python<3.10
from __future__ import annotations

from dataloaders import DataLoader
from gpt import GPT, GPTGenerator
from tqdm import tqdm
import logging
from configs import GPTConfig, GPTDataConfig, GPTTrainConfig, OptimizerConfig, Config
import torch
import time
import wandb
import math
import inspect
import tiktoken
import os
import sys


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import log, IS_DDP_RUN, RANK, WORLD_SIZE, LOCAL_RANK, IS_MASTER


def set_logging_params() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


if IS_DDP_RUN:
    dist.init_process_group(backend="nccl")
    assert torch.cuda.is_available(), "CUDA must be available for ddp run"


def log_wandb(*args, **kwargs):
    if IS_MASTER:
        wandb.log(*args, **kwargs)


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


def count_params(model):
    param_count = sum([p.nelement() for p in model.parameters()])
    return param_count


def get_memory_size(model):
    n_bytes = sum([p.nelement() * p.element_size() for p in model.parameters()])
    return n_bytes


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
    log(
        f"Using weight decay of {weight_decay} for {len(decay_params)} tensors ({sum([p.numel() for p in decay_params])} parameters)"
    )
    log(
        f"Using no weight decay for {len(nodecay_params)} tensors ({sum([p.numel() for p in nodecay_params])} parameters)"
    )
    use_fused = (
        "cuda" in device and "fused" in inspect.signature(torch.optim.AdamW).parameters
    )
    if use_fused:
        log("Using fused version of the optimizer")
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=1e-8,
        betas=optimizer_config.betas,
        eps=optimizer_config.eps,
        fused=use_fused,
    )
    return optimizer


def optimize_step(model, optimizer, clip_grad_max_norm, lr):
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=clip_grad_max_norm
    )
    for g in optimizer.param_groups:
        g["lr"] = lr
    optimizer.step()
    return gradient_norm


def train_step(model, data_loader, gradient_accum_batch_size, device_type, device):
    total_loss = 0.0
    for microstep in range(gradient_accum_batch_size):
        x, y = data_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss /= gradient_accum_batch_size
        total_loss += loss.detach()
        if IS_DDP_RUN:
            model.require_backward_grad_sync = (
                microstep == gradient_accum_batch_size - 1
            )
        loss.backward()
    if IS_DDP_RUN:
        dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
    return total_loss


@torch.no_grad()
def eval_step(
    val_loader: DataLoader, device: str, device_type: str, gpt, val_steps: int
):
    val_loader.reset()
    total_val_loss = 0.0
    steps = range(val_steps)
    if IS_MASTER:
        steps = tqdm(steps)
    for _ in steps:
        x, y = val_loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, val_loss = gpt(x, y)
        total_val_loss += val_loss.detach()
    if IS_DDP_RUN:
        dist.all_reduce(total_val_loss, op=dist.ReduceOp.AVG)
    avg_loss = total_val_loss / val_steps
    return avg_loss


def train(USE_WANDB=False):
    data_config = GPTDataConfig()
    train_config = GPTTrainConfig()
    optimizer_config = OptimizerConfig()
    model_config = GPTConfig(vocab_size=50304)
    config = Config(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
        optimizer_config=optimizer_config,
    )
    if USE_WANDB and IS_MASTER:
        wandb.init(project="gpt2", name=train_config.run_name, config=config)
    device = detect_device()
    if train_config.seed is not None:
        torch.manual_seed(train_config.seed + RANK)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(train_config.seed + RANK)

    logging.info(f"Using {device=}")

    torch.set_float32_matmul_precision(train_config.float32_matmul_precision)

    tokenizer = tiktoken.get_encoding("gpt2")
    # gpt = GPT.from_pretrained("gpt2")
    gpt = GPT(model_config)
    gpt.to(device)
    log("compiling torch model...")
    gpt = torch.compile(gpt)
    log(f"{gpt.config=}")
    micro_batch_size = train_config.micro_batch_size
    train_loader = DataLoader(
        path=data_config.path,
        batch_size=micro_batch_size,
        block_size=gpt.config.block_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        split="train",
        limit_files=data_config.limit_files,
    )
    val_loader = DataLoader(
        path=data_config.path,
        batch_size=micro_batch_size,
        block_size=gpt.config.block_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        split="val",
    )

    gpt_generator = GPTGenerator(gpt, tokenizer, device)

    if IS_DDP_RUN:
        gpt = DDP(gpt, device_ids=[LOCAL_RANK])

    assert (
        train_config.tokens_per_batch
        % (model_config.block_size * train_config.micro_batch_size * WORLD_SIZE)
        == 0
    )
    gradient_accum_batch_size = train_config.tokens_per_batch // (
        model_config.block_size * train_config.micro_batch_size * WORLD_SIZE
    )
    log(f"will use {gradient_accum_batch_size} gradient accumulation steps")
    log(f"found {count_params(gpt)} parameters")
    log(f"parameters size ~ {get_memory_size(gpt) / 1024 / 1024:.2f} MB")

    n_steps = optimizer_config.max_steps
    log(f"Will train for {n_steps} steps")
    optimizer = get_optimizer(optimizer_config, gpt, device=device)
    device_type = "cuda" if "cuda" in device else device
    for step in range(n_steps):
        time0 = time.time()
        gpt.train()
        optimizer.zero_grad()
        total_loss = train_step(
            model=gpt,
            data_loader=train_loader,
            gradient_accum_batch_size=gradient_accum_batch_size,
            device=device,
            device_type=device_type,
        )

        lr = get_lr_for_step(optimizer_config, step=step)
        gradient_norm = optimize_step(
            clip_grad_max_norm=optimizer_config.clip_grad_max_norm,
            model=gpt,
            optimizer=optimizer,
            lr=lr,
        )
        torch.cuda.synchronize()
        time1 = time.time()

        total_time = time1 - time0
        throughput = train_config.tokens_per_batch / total_time
        log_payload = dict(
            lr=lr, loss=total_loss, throughput=throughput, gradient_norm=gradient_norm
        )
        log(
            f"step {step:3d} took {total_time*1000:.2f} millis | "
            + " | ".join(
                [
                    f"{k}={v:.4e}" if k == "lr" else f"{k}={v:.2f}"
                    for k, v in log_payload.items()
                ]
            )
        )
        if USE_WANDB:
            log_wandb(log_payload)

        if step % train_config.val_interval == 0:
            # eval
            log("Evaluating the model...")
            gpt.eval()
            val_loss = eval_step(
                val_loader,
                device,
                device_type,
                gpt,
                val_steps=train_config.val_microbatch_steps,
            )
            log(f"step={step} | val_loss={val_loss.item()}")
            if USE_WANDB:
                log_wandb({"val_loss": val_loss.item()})

        if step % train_config.generate_interval == 0:
            # generate
            seed_text = "Hello, I am a language model,"
            if IS_MASTER:
                gpt.eval()
                log("Generateing sample texts")
                print("-" * 100)
                torch.manual_seed(42)
                for sentence in gpt_generator.generate(seed_text, 50, 5):
                    print(sentence)
                    print()
                print("-" * 100)
        if step % train_config.checkpoint_interval == 0:
            checkpoints_dir = "checkpoints"
            os.makedirs(checkpoints_dir, exist_ok=True)
            checkpoint_path = f"{checkpoints_dir}/checkpoint_{step:06}.pt"
            log(f"saving checkpoint to {checkpoint_path}")
            torch.save(gpt.state_dict(), checkpoint_path)

    # save the model
    torch.save(gpt.state_dict(), "state_dict.pt")


if __name__ == "__main__":
    set_logging_params()
    train(USE_WANDB=True)
    if IS_DDP_RUN:
        dist.destroy_process_group()
