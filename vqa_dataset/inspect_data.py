from datasets import load_dataset
import aiohttp

# dataset = load_dataset("Graphcore/vqa", split="validation", trust_remote_code=True)
# dataset = load_dataset("Graphcore/vqa", split="train", trust_remote_code=True)

train_dataset = load_dataset(
    "Graphcore/vqa",
    split="train",
    trust_remote_code=True,
    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
)
# print(train_dataset)

val_dataset = load_dataset(
    "Graphcore/vqa",
    split="validation",
    trust_remote_code=True,
    storage_options={'client_kwargs': {'timeout': aiohttp.ClientTimeout(total=3600)}}
)
# print(val_dataset)

for i in range(100):
    if len(train_dataset[i]['label']['ids']) > 1:
        print(train_dataset[i])