from dataset.text_dataset import TextDataset
from dataset.captions_dataset import CaptionsDataset
from transformer import TransformerVLM, count_parameters

import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.functional as F
import time
import math
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", "cuda" if torch.cuda.is_available() else "cpu")

MAX_SEQUENCE_LENGTH = 2048  # Max sequence length for MiniVLM

#region Load text Dataset
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
print(f"Text datasets loaded: 5B tokens in training set and 50M tokens in validation set.")
#endregion

#region Load Captions Dataset
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
print(f"Captions datasets loaded: {len(train_captions_dataset)} training samples and {len(val_captions_dataset)} validation samples.")
#endregion

#region DataLoaders
text_batch_size = 2
captions_batch_size = 256 # Using larger batch size because these examples are shorter (in terms of toknens)

num_cpu_cores = len(os.sched_getaffinity(0))
print(f"Detected {num_cpu_cores} CPU cores.")

text_collate_fn = lambda batch: (batch, None)  # batch is a list of text tokens, None stands for no images
train_text_dataloader = DataLoader(
    train_text_dataset,
    batch_size=text_batch_size,
    num_workers=0,                          # IterableDataset often better with 0 workers
    collate_fn=text_collate_fn
)
val_text_dataloader = DataLoader(
    val_text_dataset,
    batch_size=text_batch_size,
    num_workers=0,                          # IterableDataset often better with 0 workers
    collate_fn=text_collate_fn
)

# {"image": img, "caption_ids": cap_ids, "index": true_idx} <----- this is an element of CaptionsDataset
captions_collate_fn = lambda batch: ([b["caption_ids"] for b in batch], [[b["image"]] for b in batch])
train_captions_dataloader = DataLoader(
    train_captions_dataset,
    batch_size=captions_batch_size,
    shuffle=True,
    num_workers=num_cpu_cores,
    collate_fn=captions_collate_fn
)
val_captions_dataloader = DataLoader(
    val_captions_dataset,
    batch_size=captions_batch_size,
    shuffle=False,
    num_workers=num_cpu_cores,
    collate_fn=captions_collate_fn
)
#endregion

#region Training hyperparameters

train_tokens = 50_000_000  # 50m tokens (for testing) ---> 5_000_000_000 for full training
eval_tokens = 100_000  # 100k tokens (for testing)  ---> 50_000_000 for full validation
eval_steps = train_tokens//20 # Evaluate ~20 times during training

#endregion

#region Load the Model

model = TransformerVLM(
    img_size=336,  # Image size for MiniVLM
    patch_size=16, 
    n_embd=1024, 
    n_layer=26, 
    n_head=16, 
    vocab_size=16000,
    max_context_length=MAX_SEQUENCE_LENGTH, 
)
print('!' * 30)
print(f"Model initialized with {count_parameters(model):,} parameters.")
print('!' * 30)

model.to(device)
model.train()

#endregion

#region Optimizer and Scheduler

base_lr      = 3e-4
min_lr       = 0.1*base_lr  # 3e-5
weight_decay = 0.1
accum_steps  = 8            # gradient accumulation to reach your effective batch
max_steps    = int(train_tokens/(text_batch_size*MAX_SEQUENCE_LENGTH*accum_steps))   # Approximate number of steps for training (train_tokens/tokens_per_step)
warmup_steps = int(max_steps * 0.1)                                             # 10% of total steps for warmup
grad_clip    = 1.0

# Param groups for weight decay (decay for all parameters except biases and layer norms)
def build_param_groups(m: nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            # print("not requires grad:", n)
            continue
        if p.ndim == 1 or n.endswith(".bias") or "norm" in n.lower():
            # print("no decay:         ", n)
            no_decay.append(p)
        else:
            # print("decay:            ", n)
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

# test the scheduler
def test_scheduler():
    for step in range(0, max_steps + 1, max(1, max_steps // 30)):
        sched.step(step)
        print(f"Step {step:5d}: LR = {optim.param_groups[0]['lr']:.6f}, "
            f"Min LR = {min_lr:.6f}, "
            f"Base LR = {base_lr:.6f}, "
            f"Warmup Steps = {warmup_steps}, "
            f"Max Steps = {max_steps}")
# test_scheduler()  # Uncomment to test the scheduler


# ----- mock training loop -----
# Assume your DataLoader yields LongTensor batches: input_ids [B, T] with next-token labels
"""
for step, input_ids in enumerate(train_loader, start=1):
    input_ids = input_ids.to(device)
    with torch.autocast(device_type="cuda", dtype=dtype):
        # typical causal LM loss (shift by one)
        logits = model(input_ids)                            # [B, T, V]
        loss = F.cross_entropy(
            logits[:, :-1].contiguous().view(-1, logits.size(-1)),
            input_ids[:, 1:].contiguous().view(-1),
            ignore_index=-100  # or your PAD if you mask pads in the labels
        ) / accum_steps

    loss.backward()

    if step % accum_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        sched.step()

    if step >= max_steps:
        break
"""

#endregion

#region Helper functions

def safe_next(iterable, og_loader):
    try:
        return next(iterable)
    except StopIteration:
        # Reset the iterator if it runs out of data
        return iter(og_loader)

def nice_num(num):
    """Format a large number by adding k, M, B, T suffixes..."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}k"
    else:
        return str(num)

def nice_time(seconds):
    """Format time in seconds to a human-readable format. (eg. 1h 30m 15s)"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds // 6:.0f0}m {seconds % 60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m {seconds % 60:.0f}s"

#endregion

#region TRAINING LOOP

TOTAL_TOKENS = 0    # How many tokens processed so far (counting only the plain text dataset)
TRAINING_STEPS = 0  # Number of training steps performed
NEXT_EVAL = 1       # Next evaluation (how many evaluations were performed + 1)

# Create iterators for the datasets
text_dataset_iter = iter(train_text_dataloader)
captions_dataset_iter = iter(train_captions_dataloader)

training_start_time = time.time()
while TOTAL_TOKENS < train_tokens:
    TRAINING_STEPS += 1
    performed_update = False  # Flag to check if an update was performed
    start_time = time.time()

    # ------------ Perform training step on text dataset ----------------
    # Get a batch of text data
    text_batch, _ = safe_next(text_dataset_iter, train_text_dataloader)
    text_batch = [t.to(device) for t in text_batch]  # Move to device

    # Calculate the loss
    logits, loss = model(text_batch, None, calculate_loss=True)
    loss.backward()
    text_loss = loss.item()

    # Update TOTAL_TOKENS -- given that each sequence is of the same length
    TOTAL_TOKENS += len(text_batch) * text_batch[0].shape[0]
    # -------------------------------------------------------------------

    # ------------ Perform training step on captions dataset ------------ COMMENTED OUT FOR NOW
    # Get a batch of captions data
    captions_batch_txt, captions_batch_img = safe_next(captions_dataset_iter, train_captions_dataloader)

    # Calculate the loss
    # Don't update TOTAL_TOKENS here, as we are not counting captions tokens
    # -------------------------------------------------------------------

    # ------------ Optimizer step ----------------------------------------
    if TRAINING_STEPS % accum_steps == 0:   # Perform optimizer step every `accum_steps`
        performed_update = True
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)   # clip gradients
        optim.step()                                                    # update parameters
        optim.zero_grad(set_to_none=True)                               # reset gradients
        sched.step()                                                    # update learning rate
    # -------------------------------------------------------------------

    # ------------ Evaluation (every `eval_steps` tokens) ---------------
    if TOTAL_TOKENS >= NEXT_EVAL * eval_steps:
        # Evaluate on text dataset
            # Get a batch of text data
            # Perform evaluation

        # Evaluate on captions dataset
            # Get a batch of captions data
            # Perform evaluation

        NEXT_EVAL += 1
    # -------------------------------------------------------------------

    end_time = time.time()
    print(f"{TRAINING_STEPS}. {nice_num(TOTAL_TOKENS)}/{nice_num(train_tokens)} tokens ({(end_time-start_time):.2f} s/it; {nice_num(TOTAL_TOKENS / (end_time - training_start_time))} tok/s) - {nice_time(max((train_tokens - TOTAL_TOKENS) / (TOTAL_TOKENS / (end_time - training_start_time)),0))} left", end='')
    print(f" | Text Loss: {text_loss:.4f}", end='')
    if performed_update:
        print(f" | UPDATED with LR: {optim.param_groups[0]['lr']:.6f}", end='')
    print()
    # print format:
    # 1. 50M/5B tokens (1.23 s/it; 2.3k tok/s) - 1h 30m left | Text Loss: 2.3456


#endregion
