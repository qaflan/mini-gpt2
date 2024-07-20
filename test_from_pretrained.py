from gpt import GPT
from transformers import GPT2LMHeadModel
import torch
from train_gpt2 import count_params


def test_pretrained():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    sd_model = model.state_dict()

    recreated_model = GPT.from_pretrained("gpt2")
    sd_recreated = recreated_model.state_dict()
    # ignore bias terms
    sd_recreated = {
        k: v for k, v in sd_recreated.items() if not k.endswith(".attn.bias")
    }
    assert set(sd_model) == set(sd_recreated)
    for k in sd_model:
        v = sd_model[k]
        v1 = sd_recreated[k]
        if (
            k.endswith(".c_attn.weight")
            or k.endswith(".c_proj.weight")
            or k.endswith(".c_fc.weight")
        ):
            v1 = v1.t()
        assert torch.all(v1 == v)