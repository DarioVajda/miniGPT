
import torch
from miniGPT import AttentionLanguageModel, vocab_size, learning_rate, get_batch, device, train_iterations, eval_interval, estimate_loss

attention_model = AttentionLanguageModel().to(device)
optimizer = torch.optim.AdamW(attention_model.parameters(), lr=learning_rate)

#region Save checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    """
    Saves the current model and optimizer state, along with the epoch and loss.

    Args:
        model (torch.nn.Module): The model being trained.
        optimizer (torch.optim.Optimizer): The optimizer.
        epoch (int): Current training epoch.
        loss (float): The loss value at this checkpoint.
        filename (str): File name for saving the checkpoint.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

# Example usage inside your training loop:
# for epoch in range(num_epochs):
#     ... # training code
#     if epoch % save_interval == 0:
#         save_checkpoint(model, optimizer, epoch, current_loss, filename=f'checkpoint_epoch_{epoch}.pth')

#endregion

#region Training loop
for steps in range(train_iterations):
    # sampling a batch of data
    xb, yb = get_batch()

    if (steps % eval_interval == 0 or steps == train_iterations-1) and steps != 0:
        validation_result = estimate_loss(attention_model)
        print(f'step: {steps}, train loss: {validation_result["train"]}, val loss: {validation_result["val"]}')
        save_checkpoint(attention_model, optimizer, steps, 0, filename=f'./checkpoints/checkpoint_epoch_{steps}.pth')

    # evaluate the loss
    logits, loss = attention_model(xb, yb)
    
    if steps % 100 == 0:
        print("Step: ", steps, " Loss: ", loss.item())
        
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#endregion