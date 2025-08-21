import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
from contextlib import nullcontext

class DistributedIterWrapper(IterableDataset):
    """
    Wrap any IterableDataset and yield only the stripe belonging to
    (global rank, worker id). No changes to the underlying dataset.
    """
    def __init__(self, base_ds: IterableDataset):
        super().__init__()
        self.base_ds = base_ds
        self._epoch = 0  # optional: used to rotate shards across epochs

    def set_epoch(self, epoch: int):
        self._epoch = int(epoch)

    def _rank_world(self):
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        else:
            return 0, 1

    def __iter__(self):
        # Global DDP sharding
        rank, world_size = self._rank_world()

        # Per-worker sharding (inside each process)
        info = torch.utils.data.get_worker_info()
        worker_id = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1

        total_shards = world_size * num_workers
        # unique shard id in [0, total_shards)
        shard_id = rank * num_workers + worker_id

        # Optional: rotate shards each epoch so every rank eventually sees all stripes
        # (keeps epoch-to-epoch variation without touching the base dataset)
        start = (shard_id + self._epoch) % total_shards
        step = total_shards

        it = iter(self.base_ds)
        # Efficient striding: yields items at indices start, start+step, start+2*step, ...
        yield from islice(it, start, None, step)

# --- helper to build loaders uniformly ---
def build_ddp_iter_loader(
    base_ds: IterableDataset,
    batch_size: int,
    num_workers: int,
    collate_fn=None,
    pin_memory=True,
    persistent_workers=False
):
    wrapped = DistributedIterWrapper(base_ds)
    loader = DataLoader(
        wrapped,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return loader