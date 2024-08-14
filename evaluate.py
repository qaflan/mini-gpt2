import torch
from dataloaders import HellaSwagLoader
from tqdm import tqdm
from torch.nn import functional as F
from utils import IS_DDP_RUN, IS_MASTER
from gpt import GPT
import torch.distributed as dist


@torch.no_grad()
def get_hellaswag_prediction(model, tokens, mask, label, device=None):
    if device:
        tokens = tokens.to(device)
        mask = mask.to(device)
    logits, _ = model(tokens)
    tokens_shifted = tokens[:, 1:]
    logits_shifted = logits[:, :-1, :]
    mask_shifted = mask[:, 1:]
    n_choices = tokens.size(0)
    y = F.cross_entropy(
        logits_shifted.reshape(-1, logits_shifted.size(-1)),
        tokens_shifted.reshape(-1),
        reduction="none",
    ).view(n_choices, -1)  # loss per token
    y = (
        y * mask_shifted
    )  # keep only the tokens that are relevant (exclude context and padding)
    loss_sum = y.sum(-1)
    loss_avg = loss_sum / mask_shifted.sum(-1)
    selected = loss_sum.argmin()
    return 1 if selected == label else 0, selected.item(), loss_avg


@torch.no_grad()
def evaluate_hellaswag(loader: HellaSwagLoader, model: GPT, device=None) -> float:
    loader.reset()
    total = torch.tensor(0)
    correct = torch.tensor(0)
    if device:
        total = total.to(device)
        correct = correct.to(device)
    rng = range(loader.n)
    if IS_MASTER:
        rng = tqdm(rng, desc="Evaluating model on HellaSwag")
    for i in enumerate(rng):
        item = loader.next_batch()
        acc, _, _ = get_hellaswag_prediction(
            model=model,
            tokens=item["tokens"],
            mask=item["mask"],
            label=item["label"],
            device=device,
        )
        correct += acc
        total += 1
    if IS_DDP_RUN:
        dist.all_reduce(total, dist.ReduceOp.SUM)
        dist.all_reduce(correct, dist.ReduceOp.SUM)
    accuracy = correct / total
    return accuracy.item()
