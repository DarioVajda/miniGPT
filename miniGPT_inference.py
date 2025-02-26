import torch
from miniGPT import AttentionLanguageModel, decode, vocab_size, learning_rate

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def load_checkpoint(filename, model, optimizer=None, map_location=torch.device('cpu')):
    """
    Loads the model (and optionally optimizer) state from a checkpoint file.

    Args:
        filename (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load state into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load state into.
        map_location (torch.device): Device to map the model (e.g., CPU or GPU).

    Returns:
        dict: The loaded checkpoint dictionary.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {filename}")
    return checkpoint


attention_model = AttentionLanguageModel(vocab_size).to(device)
optimizer = torch.optim.Adam(attention_model.parameters(), lr=learning_rate)

checkpoint = load_checkpoint('./checkpoints/checkpoint_epoch_4999.pth', attention_model, optimizer)
# checkpoint = load_checkpoint('./checkpoints/checkpoint_epoch_500.pth', attention_model, optimizer)
attention_model.eval()

print(decode(attention_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 2000)[0].tolist()))