import torch
from dataloaders import HellaSwagLoader, DataLoader
from tqdm import tqdm
from torch.nn import functional as F
from utils import IS_DDP_RUN, IS_MASTER, RANK, WORLD_SIZE
from gpt import GPT
import torch.distributed as dist
from configs import GPTConfig, GPTDataConfig, GPTTrainConfig


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


if __name__ == "__main__":
    import torch.distributed as dist

    torch.set_float32_matmul_precision('high')
    if IS_DDP_RUN:
        dist.init_process_group(backend="nccl")
        assert torch.cuda.is_available(), "CUDA must be available for ddp run"
    import tiktoken
    from gpt import GPT

    if IS_DDP_RUN:
        device = f"cuda:{RANK}"
    else:
        device = "cuda:1"
    tokenizer = tiktoken.get_encoding("gpt2")
    loader = HellaSwagLoader(
        tokenizer, split="validation", rank=RANK, world_size=WORLD_SIZE
    )

    print(f"{loader.n=}")
    model = GPT(GPTConfig())#.from_pretrained("gpt2").to(device)
    data_config = GPTDataConfig()
    
    train_loader = DataLoader(
        path=data_config.path,
        batch_size=16,
        block_size=model.config.block_size,
        rank=RANK,
        world_size=WORLD_SIZE,
        split="train",
        limit_files=1,
    )
    
    model = model.to(device)
    print("compiling...")
    model = torch.compile(model)
    print("Compiled")
    optimizer = torch.optim.Adam(lr=0.001)
    optimizer.zero_grad()
    x, y = train_loader.next_batch()
    logits, loss:torch.Tensor = model(x.to(device), y.to(device))
    print(loss)
    loss.backward()
    model.eval()

    acc = evaluate_hellaswag(loader, model, device=device)
    if IS_MASTER:
        print(f"{acc=}")
