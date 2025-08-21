import os, argparse, torch, torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from contextlib import nullcontext

from dataset.text_dataset import TextDataset
from dataset.captions_dataset import CaptionsDataset, save_tensor_image
from transformer import TransformerVLM, count_parameters

from dataset.ddp_iter_shard import build_ddp_iter_loader

import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import math
from torch import nn
from tqdm import tqdm
import torchvision.transforms as T

#region CONSTANTS
MAX_SEQUENCE_LENGTH = 2048
text_batch_size = 1
captions_batch_size = 4

train_tokens = 50_000_000  # 50m tokens (for testing) ---> 5_000_000_000 for full training
eval_tokens = 50_000  # 50k tokens (for testing)  ---> 50_000_000 for full validation
eval_steps = train_tokens//25 # Evaluate ~25 times during training

base_lr      = 3e-4
min_lr       = 0.1*base_lr  # 3e-5
weight_decay = 0.1
accum_steps  = 8            # gradient accumulation to reach your effective batch
max_steps    = int(train_tokens/(text_batch_size*MAX_SEQUENCE_LENGTH*accum_steps))   # Approximate number of steps for training (train_tokens/tokens_per_step)
warmup_steps = int(max_steps * 0.1)                                             # 10% of total steps for warmup
grad_clip    = 1.0
#endregion

# Function to get the text dataset (the DDP compatible dataloader)
def get_txt_data(rank):
    train_text_dataset = TextDataset(
        txt_path="dataset/tokenized_ccnews_5b.txt",
        block_size=MAX_SEQUENCE_LENGTH-2, # BOS + 4094 tokens + EOS
        bos_id=1,
        eos_id=2,
        vocab_size=16000,
        pad_id=3
    )
    val_text_dataset = TextDataset(
        txt_path="dataset/tokenized_ccnews_50m_eval.txt",
        block_size=MAX_SEQUENCE_LENGTH-2, # BOS + 4094 tokens + EOS
        bos_id=1,
        eos_id=2,
        vocab_size=16000,
        pad_id=3
    )
    if rank == 0: print(f"Text datasets loaded: 5B tokens in training set and 50M tokens in validation set.")

    text_collate_fn = lambda batch: (batch, None) # batch is a list of text tokens, None stands for no images

    train_text_dataloader = build_ddp_iter_loader(
        train_text_dataset,
        batch_size=text_batch_size,
        num_workers=0,
        collate_fn=text_collate_fn
    )
    val_text_dataloader = build_ddp_iter_loader(
        val_text_dataset,
        batch_size=text_batch_size,
        num_workers=0,
        collate_fn=text_collate_fn
    )
    if rank == 0: print("Prepared text dataloaders for DDP training.")
    return train_text_dataloader, val_text_dataloader

# Function to get the captions dataset (the DDP compatible dataloader)
def get_captions_data(rank, world_size):
    train_captions_dataset = CaptionsDataset(
        img_dir="../captions_dataset/train",
        captions_path="dataset/tokenized_train_captions_dataset.txt",
        resize_to=336,
        train_mode=True,
        pad_id=3
    )
    val_captions_dataset = CaptionsDataset(
        img_dir="../captions_dataset/validation",
        captions_path="dataset/tokenized_val_captions_dataset.txt",
        resize_to=336,
        train_mode=False,
        pad_id=3
    )
    if rank == 0: print(f"Captions datasets loaded: {len(train_captions_dataset)} training samples and {len(val_captions_dataset)} validation samples.")

    train_sampler = DistributedSampler(train_captions_dataset, shuffle=True, drop_last=True)
    val_sampler = DistributedSampler(val_captions_dataset, shuffle=False, drop_last=False)

    num_cpu_cores = len(os.sched_getaffinity(0)) // world_size
    if rank == 0: print(f"Detected {os.sched_getaffinity(0)} CPU cores. Using {num_cpu_cores} workers per DDP process.")

    # {"image": img, "caption_ids": cap_ids, "index": true_idx} <----- this is an element of CaptionsDataset
    def captions_collate_fn(batch):
        captions = [b["caption_ids"] for b in batch if b['image'] is not None]
        images = [[b["image"]] for b in batch if b['image'] is not None]
        return (captions, images)
    train_captions_dataloader = DataLoader(
        train_captions_dataset,
        batch_size=captions_batch_size,
        num_workers=num_cpu_cores,
        collate_fn=captions_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
    )
    val_captions_dataloader = DataLoader(
        val_captions_dataset,
        batch_size=captions_batch_size,
        num_workers=num_cpu_cores,
        collate_fn=captions_collate_fn,
        sampler=val_sampler,
        pin_memory=True,
    )
    if rank == 0: print("Prepared captions dataloaders for DDP training.")

    return train_captions_dataloader, val_captions_dataloader


# Function to get the model
def get_model(local_rank, device):
    model = TransformerVLM(
        img_size=336,  # Image size for MiniVLM
        patch_size=16, 
        n_embd=1024, 
        n_layer=26, 
        n_head=16, 
        vocab_size=16000,
        max_context_length=MAX_SEQUENCE_LENGTH, 
    ).to(device)
    if local_rank == 0:
        print('!' * 60)
        print(f"Model initialized with {count_parameters(model):,} parameters.")
        print('!' * 60)

    model.train()

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    print(f"Model wrapped in DDP on device {local_rank}.")
    
    return model

# Function that returns the AdamW optimizer and a cosine scheduler with linear warmup
def get_optim_and_sched(model):
    # Param groups for weight decay (decay for all parameters except biases and layer norms)
    def build_param_groups(m: nn.Module, wd: float):
        decay, no_decay = [], []
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay,    "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0},
        ]
    
    # Using AdamW optimizer
    optim = torch.optim.AdamW(
        build_param_groups(model, weight_decay),
        lr=base_lr, betas=(0.9, 0.95), eps=1e-8,
    )

    # Linear warmup + cosine decay scheduler
    def make_cosine_with_warmup(optimizer, warmup, total, base_lr, min_lr):
        def lr_lambda(step):
            if step < warmup:
                return step / max(1, warmup)
            progress = (step - warmup) / max(1, total - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            # scale between min_lr and base_lr
            return (min_lr / base_lr) + (1 - (min_lr / base_lr)) * cosine
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    sched = make_cosine_with_warmup(optim, warmup_steps, max_steps, base_lr, min_lr)

    return optim, sched


#region Helper functions

def safe_next(it, loader):
    try:
        return next(it), it
    except StopIteration:
        it = iter(loader)
        return next(it), it

def nice_num(num):
    """Format a large number by adding k, M, B, T suffixes..."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return f"{num:.1f}"

def nice_time(seconds):
    """Format time in seconds to a human-readable format. (eg. 1h 30m 15s)"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m {seconds % 60:.0f}s"

def inspect_images(captions_batch_img, save_dir="./test_image_batch", stop_program=True):
    output_dir = Path(save_dir)
    output_dir.mkdir(exist_ok=True)
    for i, img in enumerate(captions_batch_img):
        img_path = output_dir / f"image_{i:02d}.jpg"
        save_tensor_image(img[0], img_path)

    if stop_program:
        print(f"Images saved to {output_dir}. Stopping the program for inspection.")
        exit(0)

def measure_time(func, *args, **kwargs):
    """Measure the time taken by a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

#endregion

#region Evaluation functions

def ddp_sum_scalars(vals, device, dtype=torch.float64):
    """
    vals: iterable of Python numbers (or tensors convertible to scalar)
    returns a list of Python floats after SUM reduction across all ranks.
    """
    t = torch.tensor(list(vals), device=device, dtype=dtype)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)  # in-place: every rank receives the SUM
    return [v.item() for v in t]

def autocast_ctx():
    if torch.cuda.is_available():
        # Pick a sensible dtype automatically
        use_bf16 = torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)  # new API
    return nullcontext()

@torch.no_grad()
def text_eval_prev(model, val_text_dataloader, eval_tokens, device):
    print("Evaluating text dataset...")
    text_eval_loss = 0.0
    text_eval_examples = 0
    eval_tokens_processed = 0
    for text_eval_batch, _ in tqdm(val_text_dataloader):
        # move batch to device
        text_eval_batch = [t.to(device) for t in text_eval_batch]
        text_eval_examples += 1
        # forward pass --> calculate loss
        logits, loss = model(text_eval_batch, None, calculate_loss=True)
        text_eval_loss += loss.item()

        # update processed tokens
        eval_tokens_processed += len(text_eval_batch) * text_eval_batch[0].shape[0]
        if eval_tokens_processed >= eval_tokens:
            break

    text_eval_loss /= text_eval_examples
    print("Finished evaluating text dataset.")
    return text_eval_loss

@torch.no_grad()
def text_eval(model, val_text_loader, eval_tokens, device):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank  = dist.get_rank()        if dist.is_initialized() else 0

    # If you cap by tokens, split work across ranks
    per_rank_cap = None
    if eval_tokens is not None:
        per_rank_cap = (eval_tokens + world - 1) // world  # ceil

    local_loss_sum = 0.0   # sum of (mean-loss * tokens_in_batch)
    local_tok_sum  = 0     # number of tokens that contributed to loss

    with autocast_ctx():
        for batch, _ in tqdm(val_text_loader, disable=(rank != 0), desc="val/text"):
            batch = [t.to(device, non_blocking=True) for t in batch]
            # tokens in this batch on THIS rank
            n_tok = sum(t.numel() for t in batch)

            _, loss = model(batch, None, calculate_loss=True)  # assume loss is mean over tokens
            local_loss_sum += loss.item() * n_tok
            local_tok_sum  += n_tok

            if per_rank_cap is not None and local_tok_sum >= per_rank_cap:
                break

    # Reduce across ranks → every rank gets the same totals
    loss_sum_g, tok_sum_g = ddp_sum_scalars([local_loss_sum, local_tok_sum], device=device)
    global_avg_loss = loss_sum_g / max(1, tok_sum_g)
    return global_avg_loss  # you can also return math.exp(global_avg_loss) for ppl


@torch.no_grad()
def captions_eval_prev(model, val_captions_dataloader, device):
    print("Evaluating captions dataset...")
    captions_eval_loss = 0.0
    for captions_eval_batch_txt, captions_eval_batch_img in tqdm(val_captions_dataloader):
        captions_eval_batch_txt = [c.to(device) for c in captions_eval_batch_txt]
        captions_eval_batch_img = [ [img[0].to(device)] for img in captions_eval_batch_img ]
        logits, loss = model(captions_eval_batch_txt, captions_eval_batch_img, calculate_loss=True)
        captions_eval_loss += loss.item()
    captions_eval_loss /= len(val_captions_dataloader)
    print("Finished evaluating captions dataset.")
    return captions_eval_loss

@torch.no_grad()
def captions_eval(model, val_captions_loader, device):
    world = dist.get_world_size() if dist.is_initialized() else 1
    rank  = dist.get_rank()        if dist.is_initialized() else 0

    local_loss_sum = 0.0
    local_tok_sum  = 0

    with autocast_ctx():
        for cap_txt, cap_img in tqdm(val_captions_loader, disable=(rank != 0), desc="val/caps"):
            # move to device
            cap_txt = [c.to(device, non_blocking=True) for c in cap_txt]
            cap_img = [[img[0].to(device, non_blocking=True)] for img in cap_img]

            # count ONLY text tokens that you compute loss over
            n_tok = sum(t.numel() for t in cap_txt)

            _, loss = model(cap_txt, cap_img, calculate_loss=True)  # mean over tokens
            local_loss_sum += loss.item() * n_tok
            local_tok_sum  += n_tok

    # Global reduction
    loss_sum_g, tok_sum_g = ddp_sum_scalars([local_loss_sum, local_tok_sum], device=device)
    global_avg_loss = loss_sum_g / max(1, tok_sum_g)
    return global_avg_loss

#endregion

def setup_distributed():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, dist.get_rank(), dist.get_world_size()

def main():
    # Parse arguments from the command line (kept it here just in case):
    # ap = argparse.ArgumentParser()
    # ap.add_argument("--arg1", type=int, default=5)
    # args = ap.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    print(f"[Rank {rank}] Local Rank: {local_rank}, World Size: {world_size}")
    torch.manual_seed(42)  # reproducible-ish across ranks


    # Get the text dataset
    train_text_loader, val_text_loader = get_txt_data(rank)
    text_dataset_iter = iter(train_text_loader)

    train_captions_loader, val_captions_loader = get_captions_data(rank, world_size)
    captions_dataset_iter = iter(train_captions_loader)

    # Get the model
    model = get_model(local_rank, device)

    # Get the optimizer and scheduler
    optim, sched = get_optim_and_sched(model)

    # Prepare for training
    TOTAL_TOKENS = 0    # How many tokens processed so far (counting only the plain text dataset)
    TRAINING_STEPS = 0  # Number of training steps performed
    NEXT_EVAL = 1       # Next evaluation (how many evaluations were performed + 1)
    tokens_per_batch = text_batch_size + (MAX_SEQUENCE_LENGTH + 2) * world_size

    # --------------------------- Training loop ---------------------------
    training_start_time = time.time()
    while TOTAL_TOKENS < train_tokens:
        sync_now = ((TRAINING_STEPS + 1) % accum_steps == 0)

        TRAINING_STEPS += 1
        start_time = time.time()

        # ------------ Getting the batch from text dataset ----------------
        # Get a batch of text data
        (text_batch, _), text_dataset_iter = safe_next(text_dataset_iter, train_text_loader)
        text_batch = [t.to(device) for t in text_batch]  # Move to device

        # Update TOTAL_TOKENS -- given that each sequence is of the same length
        # TOTAL_TOKENS += len(text_batch) * text_batch[0].shape[0]
        TOTAL_TOKENS += tokens_per_batch  # Approximate number of tokens processed in this step
        # -------------------------------------------------------------------

        # ------------ Getting the batch from captions dataset ------------
        # Get a batch of captions data
        (captions_batch_txt, captions_batch_img), captions_dataset_iter = safe_next(captions_dataset_iter, train_captions_loader)

        captions_batch_txt = [ c.to(device) for c in captions_batch_txt ]               # Move to device
        captions_batch_img = [ [ img[0].to(device) ] for img in captions_batch_img ]    # Move to device
        # -------------------------------------------------------------------

        # ------------ Forward pass and loss calculation -------------------
        # Calculate the loss
        ctx = nullcontext() if sync_now else model.no_sync()
        with ctx:
            logits, loss_t = model(text_batch, None, calculate_loss=True)
            logits, loss_c = model(captions_batch_txt, captions_batch_img, calculate_loss=True)

            ((loss_t+loss_c) / accum_steps).backward()

            text_loss = loss_t.item()
            captions_loss = loss_c.item()
        # -------------------------------------------------------------------

        # ------------ Optimizer step ----------------------------------------
        if sync_now:   # Perform optimizer step every `accum_steps`
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # clip gradients
            optim.step()                                                    # update parameters
            optim.zero_grad(set_to_none=True)                               # reset gradients
            sched.step()                                                    # update learning rate
        # -------------------------------------------------------------------

        # ------------ Evaluation (every `eval_steps` tokens) ---------------
        if TOTAL_TOKENS >= NEXT_EVAL * eval_steps:
            model.eval()

            if rank == 0: print(f"Starting evaluation #{NEXT_EVAL} at {nice_num(TOTAL_TOKENS)} tokens...")
            text_eval_loss = text_eval(model, val_text_loader, eval_tokens, device)
            captions_eval_loss = captions_eval(model, val_captions_loader, device)
            if rank == 0: print(f"Evaluation #{NEXT_EVAL} completed: ")
            if rank == 0: print(f"Text Loss: {text_eval_loss:.4f}, Captions Loss: {captions_eval_loss:.4f}")
            
            model.train()  

            NEXT_EVAL += 1
        # -------------------------------------------------------------------

        end_time = time.time()
        if rank == 0: print(f"{TRAINING_STEPS}. {nice_num(TOTAL_TOKENS)}/{nice_num(train_tokens)} tokens ({(end_time-start_time):.2f} s/it; {nice_num(TOTAL_TOKENS / (end_time - training_start_time))} tok/s) - {nice_time(max((train_tokens - TOTAL_TOKENS) / (TOTAL_TOKENS / (end_time - training_start_time)),0))} left", end='')
        if rank == 0: print(f" | Text Loss: {text_loss:.4f}", end='')
        if rank == 0: print(f" | Captions Loss: {captions_loss:.4f}", end='')
        if sync_now:
            if rank == 0: print(f" | UPDATED with LR: {optim.param_groups[0]['lr']:.6f}", end='')
        if rank == 0: print()
        # print format:
        # 1. 50M/5B tokens (1.23 s/it; 2.3k tok/s) - 1h 30m left | Text Loss: 2.3456 | Captions Loss: 1.2345 | UPDATED with LR: 0.000300


    dist.destroy_process_group()

if __name__ == "__main__":
    main()