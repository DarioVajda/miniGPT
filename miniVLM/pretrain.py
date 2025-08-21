from dataset.text_dataset import TextDataset
from dataset.captions_dataset import CaptionsDataset, save_tensor_image
from transformer import TransformerVLM, count_parameters

import os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time
import math
from torch import nn
from tqdm import tqdm
import torchvision.transforms as T

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
text_batch_size = 1
captions_batch_size = 4

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
def captions_collate_fn(batch):
    captions = [b["caption_ids"] for b in batch if b['image'] is not None]
    images = [[b["image"]] for b in batch if b['image'] is not None]
    return (captions, images)
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
eval_tokens = 50_000  # 50k tokens (for testing)  ---> 50_000_000 for full validation
eval_steps = train_tokens//500 # Evaluate ~100 times during training

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

#endregion

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

@torch.no_grad()
def text_eval(model, val_text_dataloader, eval_tokens):
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
def captions_eval(model, val_captions_dataloader):
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
    (text_batch, _), text_dataset_iter = safe_next(text_dataset_iter, train_text_dataloader)
    text_batch = [t.to(device) for t in text_batch]  # Move to device

    # Calculate the loss
    logits, loss = model(text_batch, None, calculate_loss=True)
    loss.backward()
    text_loss = loss.item()

    # Update TOTAL_TOKENS -- given that each sequence is of the same length
    TOTAL_TOKENS += len(text_batch) * text_batch[0].shape[0]
    # -------------------------------------------------------------------

    # ------------ Perform training step on captions dataset ------------
    # Get a batch of captions data
    (captions_batch_txt, captions_batch_img), captions_dataset_iter = safe_next(captions_dataset_iter, train_captions_dataloader)

    captions_batch_txt = [ c.to(device) for c in captions_batch_txt ]               # Move to device
    captions_batch_img = [ [ img[0].to(device) ] for img in captions_batch_img ]    # Move to device

    # Calculate the loss
    logits, loss = model(captions_batch_txt, captions_batch_img, calculate_loss=True)
    loss.backward()
    captions_loss = loss.item()
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
        model.eval()

        print(f"Starting evaluation #{NEXT_EVAL} at {nice_num(TOTAL_TOKENS)} tokens...")
        text_eval_loss = text_eval(model, val_text_dataloader, eval_tokens)
        captions_eval_loss = captions_eval(model, val_captions_dataloader)
        print(f"Evaluation #{NEXT_EVAL} completed: ")
        print(f"Text Loss: {text_eval_loss:.4f}, Captions Loss: {captions_eval_loss:.4f}")
        
        model.train()  

        NEXT_EVAL += 1
    # -------------------------------------------------------------------

    end_time = time.time()
    print(f"{TRAINING_STEPS}. {nice_num(TOTAL_TOKENS)}/{nice_num(train_tokens)} tokens ({(end_time-start_time):.2f} s/it; {nice_num(TOTAL_TOKENS / (end_time - training_start_time))} tok/s) - {nice_time(max((train_tokens - TOTAL_TOKENS) / (TOTAL_TOKENS / (end_time - training_start_time)),0))} left", end='')
    print(f" | Text Loss: {text_loss:.4f}", end='')
    print(f" | Captions Loss: {captions_loss:.4f}", end='')
    if performed_update:
        print(f" | UPDATED with LR: {optim.param_groups[0]['lr']:.6f}", end='')
    print()
    # print format:
    # 1. 50M/5B tokens (1.23 s/it; 2.3k tok/s) - 1h 30m left | Text Loss: 2.3456 | Captions Loss: 1.2345 | UPDATED with LR: 0.000300


#endregion
