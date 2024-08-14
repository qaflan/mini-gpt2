from dataclasses import dataclass, field
from typing import Tuple
from utils import IS_DDP_RUN


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # 50k BPE merges + 256 bytes tokens + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


@dataclass
class GPTTrainConfig:
    compile: bool = True
    micro_batch_size: int = 16
    tokens_per_batch: int = 524288  # 2**19 ~0.5M
    seed: int = 1337
    float32_matmul_precision: str = "high"
    val_microbatch_steps: int = 500
    val_interval: int = 300
    generate_interval: int = 300
    run_name: str = "fineweb-train-full +HellaSwag"
    checkpoint_interval: int = 200
    hellaswag_interval: int = 2

    def __post_init__(self):
        if self.compile and self.hellaswag_interval > 0 and IS_DDP_RUN:
            raise AttributeError(
                "Evaluating hellaswag with compiled models do not work with torchrun. Please set compile=False or disable hellaswag evaluation and try again."
            )


@dataclass
class OptimizerConfig:
    betas: Tuple[float, float] = (
        0.9,
        0.95,
    )
    weight_decay: float = 0.1
    eps: float = 1e-8
    clip_grad_max_norm: float = 1.0
    warmup_steps: int = 715
    max_lr: float = 6e-4
    min_lr: float = field(init=False)
    max_steps: int = 19073  # 10B tokens / tokens_per_batch

    def __post_init__(self):
        self.min_lr = self.max_lr * 0.1


@dataclass
class GPTDataConfig:
    path: str = "fineweb_edu"
    limit_files: int = 1


@dataclass
class Config:
    data_config: GPTDataConfig
    model_config: GPTConfig
    optimizer_config: OptimizerConfig
    train_config: GPTTrainConfig
