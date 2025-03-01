import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#region Hyperparameters
batch_size = 64
context_length = 256
eval_iters = 200
eval_interval = 500
train_iterations = 5000
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#endregion

#region Loading the data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#endregion

#region Creating the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
#endregion

#region Tokenization
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[ch] for ch in x]
decode = lambda x: ''.join([itos[i] for i in x])
#endregion

#region Tokenizing the dataset
data = torch.tensor(encode(text), dtype=torch.int64)
#endregion

# Train/Val split
n = int(len(data) * 0.9)
train_data, val_data = data[:n], data[n:]
#endregion

#region Data Loader and Batching
def get_batch(split='train'):
    data = train_data if split=='train' else val_data                   # choosing the data from where to take the example
    ix = torch.randint(0, len(data) - context_length, (batch_size,))    # choosing the starting index of the example
    x = torch.stack([ data[i:i+context_length] for i in ix ])           # creating the context tensor
    y = torch.stack([ data[i+1:i+context_length+1] for i in ix])        # creating the target tensor
    x, y = x.to(device), y.to(device)                                   # moving the tensors to the device
    return x, y
#endregion

#region Validation
@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean()
    model.train()
    return out
#endregion

#region Self Attention Head
class Head(nn.Module):
    """ This is one head of Self-Attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # computing the attention matrix
        weights = q @ k.transpose(-2, -1) * C**-0.5 # shape: [B, T, T]
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights) 

        # computing the output
        v = self.value(x)                           # shape: [B, T, C]
        out = weights @ v                           # shape: [B, T, C]
        return out
#endregion

#region Multi Head Attention
class MultiHeadAttention(nn.Module):
    """ This is a Multi-Head Self-Attention Layer"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
#endregion

#region Feed Forward Layer
class FeedForward(nn.Module):
    """ This is a simple Multi-Layer Perceptron"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # this is the projection layer going back into the residual pathway
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
#endregion

#region Layer Normalization
class MyLayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def __call__(self, x):
        xmean = x.mean(1, keepdim=True)
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out
    
    def parameters(self):
        return [self.gamma, self.beta]
#endregion

#region Block
class Block(nn.Module):
    """ This is a Transformer Block with Multi-Head Attention and Feed Forward Layer"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention_heads = MultiHeadAttention(num_heads=n_head, head_size=head_size)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.self_attention_heads(self.ln1(x))
        x = x + self.feed_forward(self.ln2(x))
        return x
#endregion

#region Defining the model
class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, n_embd=n_embd):
        super().__init__()
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     Block(n_embd, n_head=4),
        #     nn.LayerNorm(n_embd),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # shapes of idx and targets are [batch_size, context_size]
        tok_emb = self.token_embdding_table(idx)                                    # shape: [batch_size, context_size, embd_size]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))     # shape: [context_size, embd_size]
        x = tok_emb + pos_emb                                                       # shape: [batch_size, context_size, embd_size]
        x = self.blocks(x)                                                          # shape: [batch_size, context_size, embd_size]
        x = self.final_ln(x)                                                        # shape: [batch_size, context_size, embd_size]
        logits = self.lm_head(x)                                                    # shape: [batch_size, context_size, vocab_size]

        # F.cross_entropy expects batch_size x CHANNELS x d_2 x d_3...
        if targets != None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)    
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # shape: [batch_size, vocab_size, context_size]
        else:
            loss = None

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # for _ in range(max_new_tokens):
        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            idx_cond = idx[:, -context_length:]                     # shape: [batch_size, context_length]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]                           # shape: [batch_size, vocab_size] (selecting the last token)
            probs = F.softmax(logits, dim=-1)                   # shape: [batch_size, vocab_size]
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: [batch_size, 1]
            idx = torch.cat([idx, idx_next], dim=1)             # shape: [batch_size, context_size+1]
        return idx
    
# attention_model = AttentionLanguageModel().to(device)
# optimizer = torch.optim.AdamW(attention_model.parameters(), lr=learning_rate)
#endregion

#region Save checkpoint
# def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
#     """
#     Saves the current model and optimizer state, along with the epoch and loss.

#     Args:
#         model (torch.nn.Module): The model being trained.
#         optimizer (torch.optim.Optimizer): The optimizer.
#         epoch (int): Current training epoch.
#         loss (float): The loss value at this checkpoint.
#         filename (str): File name for saving the checkpoint.
#     """
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss
#     }
#     torch.save(checkpoint, filename)
#     print(f"Checkpoint saved to {filename}")
#endregion

#region Training loop
# for steps in range(train_iterations):
#     # sampling a batch of data
#     xb, yb = get_batch()
#     print(steps)

#     # if steps % eval_interval == 0 or steps == train_iterations-1:
#     #     validation_result = estimate_loss(attention_model)
#     #     print(f'step: {steps}, train loss: {validation_result["train"]}, val loss: {validation_result["val"]}')

#     if steps % 3 == 0:
#         save_checkpoint(attention_model, optimizer, steps, 0, filename=f'./checkpoints/checkpoint_epoch_{steps}.pth')

#     # evaluate the loss
#     logits, loss = attention_model(xb, yb)
#     print(loss)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()
#endregion

#region Generating text
# print(decode(attention_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist()))
#endregion