from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257  # 50k BPE merges + 256 bytes tokens + <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embed: int = 768


@dataclass
class GPTTrainConfig:
    batch_size: int = 18
    seed: int = 1337
    float32_matmul_precision: str = "high"
    learning_rate: float = 3e-4


@dataclass
class Config:
    model_config: GPTConfig
    train_config: GPTTrainConfig
