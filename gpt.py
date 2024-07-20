from transformers import GPT2LMHeadModel
from configs import GPTConfig
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import re
import logging

@torch.no_grad()
def set_and_check(tensor: torch.Tensor, new_data: torch.Tensor, hard=True):
    if tensor.size() != new_data.size():
        logging.error(f"Size mismatch {(tensor.size(), new_data.size())}")
        if hard:
            assert tensor.size() == new_data.size()
    tensor.copy_(new_data)


def layer_norm_from_pretrained(params):
    weight = params["weight"]
    bias = params["bias"]
    layer_norm = torch.nn.LayerNorm(weight.size(0))
    layer_norm.weight.data = weight.data
    layer_norm.bias.data = bias.data
    return layer_norm


def embedding_from_pretrained(params):
    weight = params["weight"]
    embedding = torch.nn.Embedding(weight.size(0), weight.size(1))
    embedding.weight.data = weight.data
    return embedding


class CausalSelfAttention(nn.Module):
    """multi-head attention"""

    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert config.n_embed % config.n_head == 0
        self.config = config
        self.c_attn = nn.Linear(
            config.n_embed, 3 * config.n_embed
        )  # q, k, v for all heads
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)  # This is conv1d in hf
        self.n_head = config.n_head
        self.n_embed = config.n_embed

        self.register_buffer(
            "bias",
            torch.ones(config.block_size, config.block_size)
            .tril()
            .view(1, 1, config.block_size, config.block_size),
        )

    @classmethod
    def from_pretrained(cls, config, params):
        obj = cls(config)
        set_and_check(obj.c_attn.weight, params["c_attn.weight"].t(), hard=True)
        set_and_check(obj.c_attn.bias, params["c_attn.bias"], hard=True)
        set_and_check(obj.c_proj.weight, params["c_proj.weight"].t(), hard=True)
        set_and_check(obj.c_proj.bias, params["c_proj.bias"], hard=True)
        return obj

    def forward(self, x):
        B, T, C = x.size()
        # x: B, T, C: batch_size, block_size, n_embed
        qkv = self.c_attn(x)  # (B, T, C - > B, T, 3C)
        q, k, v = qkv.split(self.n_embed, dim=2)  # q: (B,T,C), ....
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # B, n_head, T, head_size
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # B, n_head, T, head_size
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # B, n_head, T, head_size
        attention = (q @ k.transpose(-1, -2)) * (
            1.0 / math.sqrt(k.size(-1))
        )  # B, n_head, T, T
        attention = attention.masked_fill(
            self.bias[:, :, :T, :T] == 0, float("-inf")
        )  # B, n_head, T, T
        attention = attention.softmax(dim=-1) @ v  # # B, n_head, T, head_size
        y = attention.transpose(1, 2)  # B, T, n_head, head_size
        y = y.contiguous().view(B, T, -1)  # B, T, C
        y = self.c_proj(y)  # B, T, C
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.c_fc = nn.Linear(config.n_embed, 4 * config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, out_features=config.n_embed)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

    @classmethod
    def from_pretrained(cls, config, params):
        obj = cls(config)
        set_and_check(obj.c_fc.weight, params["c_fc.weight"].t(), hard=True)
        set_and_check(obj.c_fc.bias, params["c_fc.bias"], hard=True)
        set_and_check(obj.c_proj.weight, params["c_proj.weight"].t(), hard=True)
        set_and_check(obj.c_proj.bias, params["c_proj.bias"], hard=True)
        return obj


def extract_params(params, prefix):
    return {
        re.sub(f"^{prefix}.", "", k): v for k, v in params.items() if f"{prefix}" in k
    }


class Block(nn.Module):
    def __init__(self, config: GPTConfig, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (B, T, C)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x  # (B,T,C)

    @classmethod
    def from_pretrained(cls, config, params):
        obj = cls(config)
        ln_1_params = extract_params(
            params=params,
            prefix="ln_1",
        )
        obj.ln_1 = layer_norm_from_pretrained(ln_1_params)

        obj.attn = CausalSelfAttention.from_pretrained(
            config,
            extract_params(
                params=params,
                prefix="attn",
            ),
        )

        ln_2_params = extract_params(params=params, prefix="ln_2")
        obj.ln_2 = layer_norm_from_pretrained(ln_2_params)

        obj.mlp = MLP.from_pretrained(
            config, extract_params(params=params, prefix="mlp")
        )
        return obj


class GPT(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embed),  # done
                wpe=nn.Embedding(config.block_size, config.n_embed),  # done
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embed),  # done
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_length: int, decoder=None):

        for _ in range(max_length):
            logits, _ = self.forward(
                idx[:, idx.size(-1) - self.config.block_size :]
            )  # (B,T,vocab_size)
            probs = logits.softmax(dim=-1)  # (B, T, vocab_size)
            probs = probs[:, -1, :]  # (B, vocab_size)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            ix = torch.multinomial(topk_probs, 1)
            next_token = torch.gather(topk_indices, -1, ix)
            idx = torch.cat([idx, next_token], dim=1)
        return idx

    def forward(self, idx: torch.Tensor, targets=None):
        # idx: (B, T)
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)  # (T)
        pos_emb = self.transformer.wpe(pos)  # (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # (B, T, n_embed)
        x = pos_emb + tok_emb  # broadcasting happens: (B, T, n_embed)
        for block in self.transformer.h:
            x = block(x)  # (B,T,n_embed)
        x = self.transformer.ln_f(x)  # (B,T,n_embed)
        logits = self.lm_head(x)  # (B,T,vocab_size)
        loss = None
        if targets is not None:
            # loss = F.cross_entropy(logits.transpose(1, 2), targets)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        config_map = {
            "gpt2": GPTConfig(
                block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embed=768
            )
        }
        config = config_map[model_type]

        obj = cls(config)
        obj.transformer.wte = embedding_from_pretrained(
            extract_params(params=sd_hf, prefix="transformer.wte")
        )
        obj.transformer.wpe = embedding_from_pretrained(
            extract_params(params=sd_hf, prefix="transformer.wpe")
        )
        for block_idx in range(obj.config.n_layer):
            h_params = extract_params(params=sd_hf, prefix=f"transformer.h.{block_idx}")
            obj.transformer.h[block_idx] = Block.from_pretrained(config, h_params)
        obj.transformer.ln_f = layer_norm_from_pretrained(
            extract_params(sd_hf, "transformer.ln_f")
        )
        set_and_check(
            obj.lm_head.weight, extract_params(params=sd_hf, prefix="lm_head")["weight"]
        )
        return obj


class GPTGenerator:
    def __init__(self, model: GPT, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, text, max_length=30, num_return_sequences=5):
        tokens = self.tokenizer.encode(text)
        tokens = (
            torch.tensor(tokens, device=self.device)
            .repeat(num_return_sequences)
            .view(num_return_sequences, -1)
        )
        result = self.model.generate(
            tokens, max_length - tokens.size(1), decoder=self.tokenizer.decode
        )
        result_decoded = [
            self.tokenizer.decode(result_.cpu().numpy()) for result_ in result
        ]
        return result_decoded
