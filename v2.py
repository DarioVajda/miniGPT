import torch
import torch.nn as nn
from torch.nn import functional as F

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

#region Hyperparameters
batch_size = 32
context_length = 8
eval_iters = 250
eval_interval = 500
train_iterations = 5000
leraning_rate = 1e-3
n_embd = 32
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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # computing the attention matrix
        weights = q @ k.transpose(-2, -1) * C**-0.5 # shape: [B, T, T]
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)

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
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
#endregion

#region Defining the model
class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size=vocab_size, n_embd=n_embd):
        super().__init__()
        self.token_embdding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.self_attention_heads = MultiHeadAttention(num_heads=4, head_size=n_embd//4)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # shapes of idx and targets are [batch_size, context_size]
        tok_emb = self.token_embdding_table(idx)                                    # shape: [batch_size, context_size, embd_size]
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # shape: [context_size, embd_size]
        x = tok_emb + pos_emb                                                       # shape: [batch_size, context_size, embd_size]
        x = self.self_attention_heads(x)                                             # shape: [batch_size, context_size, embd_size]
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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]                     # shape: [batch_size, context_length]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]                           # shape: [batch_size, vocab_size] (selecting the last token)
            probs = F.softmax(logits, dim=-1)                   # shape: [batch_size, vocab_size]
            idx_next = torch.multinomial(probs, num_samples=1)  # shape: [batch_size, 1]
            idx = torch.cat([idx, idx_next], dim=1)             # shape: [batch_size, context_size+1]
        return idx
    
bigram_model = AttentionLanguageModel().to(device)
optimizer = torch.optim.AdamW(bigram_model.parameters(), lr=leraning_rate)
#endregion

#region Training loop
for steps in range(train_iterations):
    # sampling a batch of data
    xb, yb = get_batch()

    if steps % eval_interval == 0 or steps == train_iterations-1:
        validation_result = estimate_loss(bigram_model)
        print(f'step: {steps}, train loss: {validation_result["train"]}, val loss: {validation_result["val"]}')

    # evaluate the loss
    logits, loss = bigram_model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
#endregion

#region Generating text
print(decode(bigram_model.generate(torch.zeros((1, 1), dtype=torch.long, device=device), 300)[0].tolist()))
#endregion