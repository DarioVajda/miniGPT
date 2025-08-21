import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torchvision import transforms as T, utils as vutils
from torchvision.transforms.functional import to_pil_image


# Stats for image normalization
CLIP_MEAN = (0.48145466, 0.45782750, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

#region image utils

# Inverse of Normalize((x - mean)/std) -> x = y*std + mean
inv_normalize = T.Normalize(
    mean=[-m/s for m, s in zip(CLIP_MEAN, CLIP_STD)],
    std=[1/s for s in CLIP_STD],
)

# Save a tensor image to a file, optionally denormalizing it first.
def save_tensor_image(tensor: torch.Tensor, path: str, denorm: bool = True):
    """
    tensor: [C,H,W] or [B,C,H,W], normalized with CLIP stats after ToTensor().
    Saves the first image if a batch is provided.
    """
    x = tensor[0] if tensor.ndim == 4 else tensor
    x = x.detach().cpu()
    if denorm:
        x = inv_normalize(x)
    x = x.clamp(0.0, 1.0)              # ToTensor() expects [0,1] range
    to_pil_image(x).save(path)
#endregion



class CaptionsDataset(Dataset):
    """
    Dataset that returns {"image": FloatTensor, "caption_ids": LongTensor, "index": int}
    and **skips** caption lines whose image file is missing.

    img_dir         : directory with 0000000.jpg, 0000005.jpg, …
    captions_path   : text/tsv file, ONE token-id caption per line
    pad_id          : tokenizer <pad> id (for the collate_fn)
    """

    def __init__(
        self,
        img_dir: str | Path,
        captions_path: str | Path,
        resize_to: int = 336,
        train_mode: bool = True,
        pad_id: int = 3,
    ):
        self.img_dir = Path(img_dir)
        self.size = resize_to
        self.pad_id = pad_id
        self.train_mode = train_mode

        # ---- 1) load captions ------------------------------------------
        with open(captions_path, "r", encoding="utf-8") as f:
            self._captions: List[List[int]] = [
                [int(tok) for tok in ln.strip().split()] for ln in f
            ]

        # ---- 2) discover existing image indices ------------------------
        existing_idx = {
            int(p.stem) for p in self.img_dir.glob("*.jpg") if p.stem.isdigit()
        }

        # ---- 3) build list of valid ids (line index == file index) -----
        self.valid_ids = [idx for idx in range(len(self._captions)) if idx in existing_idx]

        if not self.valid_ids:
            raise RuntimeError("No matching image–caption pairs found!")

        print(f"Captions dataset built with {len(self.valid_ids):,} valid pairs "
              f"(skipped {len(self._captions) - len(self.valid_ids):,} missing images).")

        # ---- transform pipeline ----------------------------------------
        if train_mode:
            self.tfm = T.Compose([
                # Light spatial aug that keeps most of the scene (good for captioning/VQA)
                T.RandomResizedCrop(
                    self.size,
                    scale=(0.90, 1.0),            # keep ≥90% of the image
                    ratio=(0.95, 1.05),           # avoid big aspect changes
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True
                ),
                # Mild photometric aug; safe for color/attribute words
                T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.3),
                T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.05),  # tiny robustness boost
                # Note: intentionally no HorizontalFlip by default (left/right questions in VQA).
                # If your data never asks about left/right/text orientation, you can add:
                # T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                # Normalise to ImageNet stats (for CLIP-like models)
                T.Normalize(
                    mean=(0.48145466, 0.45782750, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])
        else:
            self.tfm = eval_transform = T.Compose([
                # Deterministic, no aug
                T.Resize(self.size, interpolation=InterpolationMode.BICUBIC, antialias=True),
                T.CenterCrop(self.size),
                T.ToTensor(),
                # Use the SAME normalization as in training:
                T.Normalize(
                    mean=(0.48145466, 0.45782750, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ])

    # -------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.valid_ids)

    def _img_path(self, idx: int) -> Path:
        return self.img_dir / f"{idx:07d}.jpg"

    def __getitem__(self, i: int) -> dict:
        true_idx = self.valid_ids[i]

        # --- image ---
        try:
            with Image.open(self._img_path(true_idx)).convert("RGB") as im:
                img = self.tfm(im)
        except Exception as e:
            print(f"!?!?!?!? Error loading image {true_idx}: {e}")
            with open("dataset/detected_bad_images.txt", "a") as f:
                print(f"{self._img_path(true_idx)}", file=f)
            return {"image": None, "caption_ids": None, "index": true_idx}

        # --- caption ---
        cap_ids = torch.tensor(self._captions[true_idx], dtype=torch.long)

        return {"image": img, "caption_ids": cap_ids, "index": true_idx}


# -----------------------------------------------------------------------
# Collate: pads captions
# -----------------------------------------------------------------------
def pad_collate(batch, pad_id: int = 3):
    imgs = torch.stack([b["image"] for b in batch], dim=0)
    caps = [b["caption_ids"] for b in batch]
    max_len = max(c.size(0) for c in caps)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, c in enumerate(caps):
        padded[i, : c.size(0)] = c
    return {
        "images": imgs,
        "caption_ids": padded,
        "lengths": torch.tensor([c.size(0) for c in caps]),
        "indices": torch.tensor([b["index"] for b in batch]),
    }

captions_batch_collate_fn = lambda b: pad_collate(b, pad_id=3)

# -----------------------------------------------------------------------
# quick smoke test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    num_cpu_cores = len(os.sched_getaffinity(0))
    print(f"Detected {num_cpu_cores} CPU cores.")

    ds = CaptionsDataset(
        img_dir="../../captions_dataset/validation",
        captions_path="tokenized_val_captions_dataset.txt",
        resize_to=336,
        train_mode=True,
        pad_id=3,
    )
    dl = DataLoader(    
        ds, batch_size=16, shuffle=True, num_workers=num_cpu_cores,
        collate_fn=lambda b: pad_collate(b, pad_id=3),
        pin_memory=True,
    )
    batch = next(iter(dl))
    print(batch["images"].shape, batch["caption_ids"].shape)

    # save images to './test_batch' directory (make the directory first)
    output_dir = Path("./test_batch")
    output_dir.mkdir(exist_ok=True)
    for i, img in enumerate(batch["images"]):
        img_path = output_dir / f"image_{i:02d}.jpg"
        img_pil = T.ToPILImage()(img)
        img_pil.save(img_path)
        print(f"Saved image {i} to {img_path}")
    print(f"Images saved to {output_dir}.")

    # tokenize the text back to verify and print to file ./test_batch/captions.txt:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file="../tokenizer/mini_vlm_sp16k.model")
    with open(output_dir / "captions.txt", "w") as f:
        for i, cap_tokens in enumerate(batch["caption_ids"]):
            cap_str = sp.decode(cap_tokens.tolist())
            captions_len = len([tok for tok in cap_tokens if tok != 3])  # count non-pad tokens
            f.write(f"Caption {i}: {cap_str} (length: {captions_len})\n")
            print(f"Caption {i}: {cap_str} (length: {captions_len})")
