# streaming_packed_tokens_dataset.py
# Document-aware streaming packing from a large tokenized text file.

import os
import io
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import IterableDataset, DataLoader

class TextDataset(IterableDataset):
    def __init__(
        self,
        txt_path: str | Path,
        block_size: int,
        bos_id: int,
        eos_id: int,
        vocab_size: Optional[int] = None,
        encoding: str = "utf-8",
        pad_id: int = 3,
        skip_empty: bool = True,
    ):
        self.txt_path = Path(txt_path)
        self.block_size = block_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.encoding = encoding
        self.pad_id = pad_id
        self.skip_empty = skip_empty

        if not self.txt_path.exists():
            raise FileNotFoundError(self.txt_path)

    def parse_line(self, line: str) -> List[int]:
        if not line and self.skip_empty:
            return []
        ids = [int(tok) for tok in line.split()]
        if self.vocab_size is not None:
            for t in ids:
                if t < 0 or t >= self.vocab_size:
                    raise ValueError(f"Token {t} out of range")
        return ids

    def line_iterator(self):
        with open(self.txt_path, "r", encoding=self.encoding) as f:
            for raw in f:
                line = raw.strip()
                if not line and self.skip_empty:
                    continue
                yield self.parse_line(line)

    def __iter__(self):
        buffer = []
        for doc in self.line_iterator():
            # Wrap with BOS/EOS
            doc = [self.bos_id] + doc + [self.eos_id]
            while doc:
                remaining_space = self.block_size - len(buffer)
                if len(doc) <= remaining_space:
                    buffer.extend(doc)
                    doc = []
                else:
                    buffer.extend(doc[:remaining_space])
                    yield torch.tensor(buffer, dtype=torch.long)
                    buffer = []
                    doc = doc[remaining_space:]

                if len(buffer) == self.block_size:
                    yield torch.tensor(buffer, dtype=torch.long)
                    buffer = []

        if buffer:
            yield torch.tensor(buffer, dtype=torch.long)


def pad_collate(batch, pad_id=3):
    B = len(batch)
    T = max(len(x) for x in batch)
    out = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for i, seq in enumerate(batch):
        out[i, :len(seq)] = seq
        attn[i, :len(seq)] = 1
    return {"input_ids": out, "attention_mask": attn}

txt_batch_collate_fn = lambda b: pad_collate(b, pad_id=3)

def pad_percentage(batch, pad_id=3):
    """
    Returns the percentage of pad tokens in the given batch.
    batch: dict with "input_ids" -> LongTensor[B, T]
    """
    input_ids = batch["input_ids"]
    total_tokens = input_ids.numel()
    pad_tokens = (input_ids == pad_id).sum().item()
    return 100.0 * pad_tokens / total_tokens


if __name__ == "__main__":
    dataset = TextDataset(
        "tokenized_ccnews_5b.txt",
        block_size=4096,
        bos_id=1,
        eos_id=2,
        vocab_size=16000,
        pad_id=3
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        collate_fn=lambda b: pad_collate(b, pad_id=3),
        num_workers=0  # IterableDataset often better with 0 workers
    )
    batch = next(iter(loader))

    # decode the first entry in the first batch
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file="../tokenizer/mini_vlm_sp16k.model")
    first_entry_tokens = batch["input_ids"][0]
    first_entry_text = sp.decode(first_entry_tokens.tolist())
    print('--' * 20)
    print(f"First entry tokens:\n{first_entry_tokens}")
    print('--' * 20)
    print(f"First entry text:\n{first_entry_text}")

    print('--' * 20)
    token_pieces = [sp.id_to_piece(t) for t in first_entry_tokens.tolist()]
    print("".join(token_pieces).replace("▁", " ").strip())