print("Checking images in the dataset...")
from captions_dataset import CaptionsDataset
print("Imported CaptionsDataset successfully.")

train_captions_dataset = CaptionsDataset(
    img_dir="../../captions_dataset/train",
    captions_path="tokenized_train_captions_dataset.txt",
    resize_to=336,
    train_mode=True,
    pad_id=3
)

# Iterate through the dataset
for i in range(len(train_captions_dataset)):
    # Check if the item will be loaded successfully
    item = train_captions_dataset[i]
    print(f"Item {i} loaded successfully")