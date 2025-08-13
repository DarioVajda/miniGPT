import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

#region Self Attention Head
class Head(nn.Module):
    """ This is one head of Self-Attention"""

    def __init__(self, head_size, n_embd, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        # self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_padding_mask=None):
        """
            x: shape [B, T, C] where B is batch size, T is sequence length, C is embedding size
            attn_padding_mask: shape [B, T] where B is batch size, T is sequence length (1 for valid token, 0 for padding token)

            Returns: shape [B, T, C] where B is batch size, T is sequence length, C is embedding size
        """
        B, T, C = x.shape
        k = self.key(x)                                # [B, T, H]
        q = self.query(x)                              # [B, T, H]

        # attention logits
        weights = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)   # [B, T, T]

        # causal mask (upper triangle j>i)
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        weights = weights.masked_fill(causal, float('-inf'))

        # padding mask on KEYS (mask columns where key is PAD)
        if attn_padding_mask is not None:
            key_pad = (attn_padding_mask == 0)         # [B, T] True where PAD
            weights = weights.masked_fill(key_pad[:, None, :], float('-inf'))  # broadcast to [B, T, T]

        # softmax + dropout
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        v = self.value(x)                              # [B, T, H]
        out = weights @ v                              # [B, T, H]

        # (optional) zero outputs at PAD query rows
        if attn_padding_mask is not None:
            out = out * attn_padding_mask[:, :, None].to(out.dtype)

        return out
#endregion

#region Multi Head Attention
class MultiHeadAttention(nn.Module):
    """ This is a Multi-Head Self-Attention Layer"""

    def __init__(self, num_heads, head_size, n_embd, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_padding_mask=None):
        out = torch.cat([head(x, attn_padding_mask=attn_padding_mask) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
#endregion

#region Feed Forward Layer
class FeedForward(nn.Module):
    """ This is a simple Multi-Layer Perceptron"""

    def __init__(self, n_embd, dropout):
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


#region Block
class Block(nn.Module):
    """ This is a Transformer Block with Multi-Head Attention and Feed Forward Layer"""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention_heads = MultiHeadAttention(num_heads=n_head, head_size=head_size, n_embd=n_embd, dropout=dropout)
        self.feed_forward = FeedForward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x, attn_padding_mask=None):
        x = x + self.self_attention_heads(self.ln1(x), attn_padding_mask=attn_padding_mask)
        x = x + self.feed_forward(self.ln2(x))
        return x
#endregion

#region Defining the model
class TransformerVLM(nn.Module):
    def __init__(
        self, 
        img_size=336, 
        patch_size=16, 
        n_embd=512, 
        n_layer=6, 
        n_head=8, 
        vocab_size=16_000, 
        max_context_length=2048, 
        dropout=0.15,
        bos_token=1,
        eos_token=2,
        pad_token=3,
        img_start_token=8, 
        img_end_token=9,
    ):
        """
            Initializes the Vision TransformerVLM model.

            Args:
                img_size (int): Size of the input images (assumed square).
                patch_size (int): Size of the patches to be extracted from the images.
                n_embd (int): Dimension of the embedding space.
                n_layer (int): Number of transformer blocks.
                n_head (int): Number of attention heads in each block.
                vocab_size (int): Size of the vocabulary for text tokens.
                max_context_length (int): Maximum length of the context for text tokens.
                dropout (float): Dropout rate for the model.
                bos_token (int): Token ID for the beginning of sequence.
                eos_token (int): Token ID for the end of sequence.
                pad_token (int): Token ID for padding.
                img_start_token (int): Token ID for the start of image tokens.
                img_end_token (int): Token ID for the end of image tokens.
        """
        super().__init__()

        #region General
        self.n_embd = n_embd

        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.final_ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.bos_token = bos_token
        self.eos_token = eos_token
        self.img_start_token = img_start_token
        self.img_end_token = img_end_token
        self.pad_token = pad_token
        #endregion

        #region Text embedding
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)           # shape: [vocab_size, embd_size]
        self.txt_pos_embeddings = nn.Embedding(max_context_length, n_embd)      # shape: [context_size, embd_size]
        self.context_length = max_context_length
        #endregion
        
        #region Image embedding
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_count = (img_size // patch_size) ** 2

        self.patch_embedding = nn.Conv2d(in_channels=3, out_channels=n_embd, kernel_size=patch_size, stride=patch_size)
        self.img_pos_embeddings = nn.Embedding(self.patch_count, n_embd)
        #endregion

    def forward(self, text_tokens_list, images, calculate_loss=False):
        """
            text_tokens_list: list with batch_size elements, each is a tensor of shape [context_size] containing the indices of the tokens
            images: list (batch_size elements) of lists (any number of elements) of images, each image is a tensor of shape [3, img_size, img_size]
            targets: tensor of shape [batch_size, context_size] containing the target indices for the text tokens (optional)
        """
        # B, T = text_tokens_list.shape
        B = len(text_tokens_list)  # batch size
        
        # Get the device on which the model weights are located
        dev = next(self.parameters()).device

        # -----------------------------------------------------
        # ------------------ Text Embedding -------------------
        # -----------------------------------------------------
        # This will be a list with tensors representing the tensor embeddings together with all special tokens
        txt_emb_list = []
        bos_emb = self.token_embedding_table(torch.tensor(self.bos_token, device=dev)).unsqueeze(0)              # shape: [1, embd_size]
        eos_emb = self.token_embedding_table(torch.tensor(self.eos_token, device=dev)).unsqueeze(0)              # shape: [1, embd_size]
        # img_wrapper = torch.cat([start_img_emb, end_img_emb], dim=0)                                                # shape: [2, embd_size]
        for i in range(B):
            seq_tokens = text_tokens_list[i]                                                    # shape: [curr_seq_len, embd_size]
            seq = self.token_embedding_table(seq_tokens)                                        # shape: [curr_seq_len, embd_size]
            # curr_seq_len = seq.shape[0]
            # Add start and end tokens for text, and image start and end tokens for each image
            # seq = torch.cat(
            #     # [bos_emb] + ([img_wrapper] * len(images[i])) + [seq, eos_emb], dim=0
            #     [ bos_emb, seq, eos_emb ], dim=0
            # )

            # Append to the list
            txt_emb_list.append(seq)                                                     # list of tensors (batch_size elements), each tensor

        # for el in txt_emb_list:
        #     print(f"Text Embedding shape: {el.shape}")  # Debugging line to check the shape of text embeddings
        # print('-' * 50)
        
        # -----------------------------------------------------
        # ------------------ Image Embedding ------------------
        # -----------------------------------------------------
        # If no images are provided, create empty lists for each batch
        if images is None: images = [[] for _ in range(B)]
        # Convert list of images to a tensor
        non_empty = [torch.stack([img.to(dev) for img in batch]) for batch in images if len(batch) > 0]

        if len(non_empty) == 0:
            x_img = [[] for _ in range(B)]
        else:
            # Concatenate all images in the batch into a single tensor
            img_tensor = torch.cat(non_empty, dim=0)                                # shape: [batch_size, n_images, 3, img_size, img_size]

            # Mapping from images to batch indexes
            counts = torch.tensor([len(x) for x in images], device=dev)
            
            img_to_batch_idx = torch.repeat_interleave(
                torch.arange(len(images), device=dev), counts
            ).view(-1)  # Flatten the tensor to match the image tensor shape
            
            # Embedding all of the images in the batch
            img_emb = self.patch_embedding(img_tensor)                                      # shape: [n_images, n_embd, H/patch_size, W/patch_size]
            img_emb = img_emb.flatten(2).transpose(1, 2)                                    # shape: [n_images, patch_count, n_embd]
            
            # Get positional embeddings for images
            img_pos_emb = self.img_pos_embeddings(
                torch.arange(self.patch_count, device=dev)                               # shape: [patch_count, n_embd]
            )
            
            # Add positional embeddings to the image embeddings
            x_img_tensor = img_emb + img_pos_emb                                            # shape: [n_images, patch_count, n_embd]

            # Get the start and end image token embeddings
            start_img_emb = self.token_embedding_table(torch.tensor(self.img_start_token, device=dev)).unsqueeze(0)  # shape: [1, embd_size]
            end_img_emb = self.token_embedding_table(torch.tensor(self.img_end_token, device=dev)).unsqueeze(0)      # shape: [1, embd_size]

            # print("x_img_tensor shape before adding special tokens:", x_img_tensor.shape)  # Debugging line to check the shape of image embeddings

            # Add start and end tokens around each image embedding
            x_img_tensor = torch.cat(
                [start_img_emb.repeat(x_img_tensor.shape[0], 1, 1)] +
                [x_img_tensor] +
                [end_img_emb.repeat(x_img_tensor.shape[0], 1, 1)],
                dim=1
            )

            # print("x_img_tensor with special tokens shape:", x_img_tensor.shape)  # Debugging line to check the shape of image embeddings
            # print("img_to_batch_idx:", img_to_batch_idx)
            
            # Arrange the image embeddings to match the batch indexes
            x_img = [[] for _ in range(B)]
            for i in range(img_to_batch_idx.shape[0]):
                x_img[img_to_batch_idx[i]].append(x_img_tensor[i]) 
            # for i, prompt_imgs in enumerate(x_img):
            #     for img in prompt_imgs:
            #         print(f"Image {img.shape} in prompt {i}")
            # print('-' * 50)

        # -----------------------------------------------------
        # --------------- Inject Image tokens -----------------
        # -----------------------------------------------------
        # Go through each prompt and append the corresponding image embeddings
        x_both_list = [ prompt for prompt in txt_emb_list ]  # Initialize the list with text embeddings
        for i in range(B):
            if len(x_img[i]) > 0:
                parts = [bos_emb] + x_img[i] + [txt_emb_list[i]]
            else:
                parts = [bos_emb, txt_emb_list[i]]
            if calculate_loss:
                parts.append(eos_emb)
            x_both_list[i] = torch.cat(parts, dim=0)
        
        #     print(f"Prompt {i} shape after adding images: {x_both_list[i].shape}")  # Debugging line to check the shape of combined embeddings
        # print('-' * 50)

        # -----------------------------------------------------
        # -------- Add per-token positional embeddings --------
        # -----------------------------------------------------
        for i in range(B):
            # Check if the prompt has more than max_context_length tokens
            if x_both_list[i].shape[0] > self.context_length:
                raise ValueError(f"Prompt {i} exceeds max context length: {x_both_list[i].shape[0]} > {self.context_length}")
            # Get positional embeddings for the current prompt
            pos_emb = self.txt_pos_embeddings(
                torch.arange(x_both_list[i].shape[0], device=dev) # shape: [curr_context_length, embd_size]
            )
            # Add positional embeddings to the image+text embeddings
            x_both_list[i] = x_both_list[i] + pos_emb

        # print("Applied positional embeddings to all prompts.")
        # print("Current x_both_list len:", len(x_both_list))  # Debugging line to check the length of the list
        # for i, x in enumerate(x_both_list):
        #     print(f"Prompt {i} shape after positional embeddings: {x.shape}")
        # print('-' * 50)

        # -----------------------------------------------------
        # ------------------ Apply padding --------------------
        # -----------------------------------------------------
        pad_token_emb = self.token_embedding_table(torch.tensor(self.pad_token, device=dev)).unsqueeze(0)  # shape: [1, embd_size]
        max_seq_len = max([x.shape[0] for x in x_both_list])  # Find the maximum sequence length in the batch

        # Pad each sequence to the maximum length and stack them into a tensor
        x_both = torch.stack([
            torch.cat([x, pad_token_emb.repeat(max_seq_len - x.shape[0], 1)], dim=0) if x.shape[0] < max_seq_len else x
            for x in x_both_list
        ], dim=0)

        # Create attention mask for padding tokens
        attn_padding_mask = torch.stack([
            torch.cat([torch.ones(x.shape[0], device=dev), torch.zeros(max_seq_len - x.shape[0], device=dev)])
            for x in x_both_list
        ])

        # Inspect the data
        # print("Padded x_both shape:", x_both.shape)
        # print("Attention padding mask shape:", attn_padding_mask.shape)  # Debugging line to check the shape of the attention mask
        # print('-' * 50)

        # -----------------------------------------------------
        # ------------------ Forward pass ---------------------
        # -----------------------------------------------------
        for block in self.blocks:
            x_both = block(x_both, attn_padding_mask=attn_padding_mask)         # shape: [batch_size, max_seq_len, embd_size]
        x = self.final_ln(x_both)                                               # shape: [batch_size, max_seq_len, embd_size]
        logits = self.lm_head(x)                                                # shape: [batch_size, max_seq_len, vocab_size]

        # -----------------------------------------------------
        # ---------------- Loss Calculation -------------------
        # -----------------------------------------------------
        if not calculate_loss:
            return logits, None

        # each batch is consisted of the following tokens:
        # bos, [img_start, ...IMG1, img_end, img_start, ...IMG2, img_end,] ...TEXT_TOKENS, eos, ...PAD
        #  0         0         0        0        0          0        0          1           1      0
        # Above are the masks (for which token positions should be predicted)
        # We shift everything by one to the left, so that we have a mask indicating which tokens predict the targets

        lens_without_pad = [ x.shape[0] for x in x_both_list ]
        # for i in range(B):
        #     print(f"lens_without_pad[{i}] =", lens_without_pad[i])
        #     print(f"text_tokens_list[{i}].shape[0] =", text_tokens_list[i].shape[0])
        #     print("x_both.shape[1] =", x_both.shape[1])
        #     print("0's --> ", lens_without_pad[i] - text_tokens_list[i].shape[0] - 1)
        #     print("1's --> ", text_tokens_list[i].shape[0] + 1)
        #     print("0's --> ", x_both.shape[1] - lens_without_pad[i])
        #     print("_-" * 25)
        txt_tokens_map = torch.stack(
            [ 
                torch.cat([ 
                    torch.zeros(lens_without_pad[i] - text_tokens_list[i].shape[0] -1), # BOS and Image tokens
                    torch.ones(text_tokens_list[i].shape[0] + 1),                       # Text tokens (+1 for EOS)
                    torch.zeros(x_both.shape[1] - lens_without_pad[i])                  # PAD tokens
                ], dim=0)
                for i in range(B)
            ],
            dim=0
        )
        # Shift to left by one
        txt_tokens_map = torch.cat(
            [txt_tokens_map[:, 1:], torch.zeros((B, 1))], dim=1
        )

        # Convert masks to bool
        txt_tokens_map = txt_tokens_map.bool()  # shape: [batch_size, max_seq_len]

        # Get probability distribution for the target token positions
        predicting_logits_list = []
        target_tokens_list = []

        for i in range(B):
            predicting_logits_list.append(logits[i][txt_tokens_map[i]])     # shape: [num_predicting_tokens, vocab_size]
            # append the tokens from input + eos token
            target_tokens_list.append(torch.cat(
                [text_tokens_list[i], torch.tensor([self.eos_token], device=dev)]
            ))  # shape: [num_predicting_tokens]

        predicting_logits = torch.cat(predicting_logits_list, dim=0)        # shape: [batch_size, num_predicting_tokens, vocab_size]
        # print("predicting_logits shape:", predicting_logits.shape)  # Debugging line to check the shape of predicting logits

        target_tokens = torch.cat(target_tokens_list, dim=0)                # shape: [batch_size, num_predicting_tokens]
        # print("target_tokens shape:", target_tokens.shape)  # Debugging line to check the shape of target tokens

        # Calculate the loss
        if target_tokens.shape[0] > 0:  # Check if there are any target tokens
            loss = F.cross_entropy(predicting_logits, target_tokens)

        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        text_tokens_list,
        images,
        max_new_tokens,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        stop_on_eos: bool = True,
    ):
        """
        Autoregressive generation conditioned on images.
        - `text_tokens_list`: list[tensor(len_i,)] of starting text (no BOS/EOS)
        - `images`: list[list[Tensor(3,H,W)]], same structure as training
        Produces new token IDs appended to each sample (returned as a list of tensors).
        """
        dev = next(self.parameters()).device
        B = len(text_tokens_list)
        texts = [t.clone().to(dev) for t in text_tokens_list]
        imgs = images if images is not None else [[] for _ in range(B)]
        finished = [False] * B

        # helper for per-sample last index (no EOS during inference)
        def last_token_index(i: int) -> int:
            nimg = len(imgs[i])
            # sequence = [BOS] + sum_i([IMG_START, patches..., IMG_END]) + text
            seq_len = 1 + nimg * (self.patch_count + 2) + texts[i].shape[0]
            return seq_len - 1  # index of last real token

        # context budget check per sample (we stop that sample if it would overflow)
        def max_text_len_allowed(i: int) -> int:
            nimg = len(imgs[i])
            # keep: 1 for BOS, 2 per image (start/end), patch_count per image
            overhead = 1 + nimg * (self.patch_count + 2)
            return self.context_length - overhead  # no EOS at inference

        for _ in tqdm(range(max_new_tokens), desc="Generating tokens"):
            if all(finished):
                break

            # stop samples that would overflow context window
            for i in range(B):
                if not finished[i] and texts[i].shape[0] >= max_text_len_allowed(i):
                    finished[i] = True

            logits, _ = self(texts, imgs, calculate_loss=False)  # NO EOS inside

            for i in range(B):
                if finished[i]:
                    continue

                idx_last = last_token_index(i)
                logits_i = logits[i, idx_last, :]

                # sampling temperature
                if temperature is not None and temperature > 0.0:
                    logits_i = logits_i / temperature

                probs = F.softmax(logits_i, dim=-1)

                # top-k
                if top_k is not None and top_k > 0:
                    v, ix = torch.topk(probs, k=min(top_k, probs.numel()))
                    mask = torch.zeros_like(probs)
                    mask.scatter_(0, ix, v)
                    probs = mask / mask.sum()

                # top-p (nucleus)
                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=0)
                    # keep smallest set whose cumulative >= top_p
                    cutoff = cum > top_p
                    # ensure at least the highest-prob token stays
                    cutoff[0] = False
                    sorted_probs[cutoff] = 0
                    probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                    probs = probs / probs.sum()

                next_id = torch.multinomial(probs, num_samples=1)

                # stop on EOS per-sample
                if stop_on_eos and int(next_id) == self.eos_token:
                    finished[i] = True
                else:
                    texts[i] = torch.cat([texts[i], next_id], dim=0)

        return texts
#endregion

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def run_generate_smoke_tests(device):
    import torch
    import torch.nn as nn

    torch.manual_seed(42)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # small, fast model
    model = TransformerVLM(
        img_size=32,       # small so patch_count=(32/16)^2=4
        patch_size=16,
        n_embd=64,
        n_layer=2,
        n_head=4,
        vocab_size=100,
        max_context_length=64,
    ).to(dev).eval()

    # ---------- Test 1: basic growth & context budget respect ----------
    B = 3
    texts = [
        torch.randint(0, 100, (5,), device=dev),   # short
        torch.randint(0, 100, (10,), device=dev),  # medium
        torch.randint(0, 100, (0,), device=dev),   # empty text
    ]
    images = [
        [torch.randn(3, 32, 32, device=dev) for _ in range(2)],  # 2 images
        [],                                                      # no images
        [torch.randn(3, 32, 32, device=dev)],                    # 1 image
    ]
    start_lens = [t.numel() for t in texts]

    out = model.generate(
        texts, images,
        max_new_tokens=5,
        temperature=0.8,
        top_p=0.95,
        stop_on_eos=False,
    )
    end_lens = [t.numel() for t in out]

    def allowed_len(model, nimg):
        # allowed text length at inference (no EOS appended):
        # context_length - (1 for BOS) - nimg*(patch_count + 2 for img_start/end)
        return model.context_length - (1 + nimg * (model.patch_count + 2))

    for i in range(B):
        max_allowed = allowed_len(model, len(images[i]))
        assert end_lens[i] <= min(start_lens[i] + 5, max_allowed), f"sample {i}: grew too much"
        assert end_lens[i] >= start_lens[i], f"sample {i}: did not grow"
    print("Test 1 (basic growth) OK.")

    # ---------- Test 2: EOS stop (force EOS via bias) ----------
    # Make EOS the most likely token everywhere (greedy/top_k=1 picks it)
    with torch.no_grad():
        if model.lm_head.bias is None:
            model.lm_head.bias = nn.Parameter(torch.zeros(model.lm_head.out_features, device=dev))
        model.lm_head.bias.zero_()
        model.lm_head.bias[model.eos_token] = 50.0  # huge bias to dominate

    texts2 = [t.clone() for t in texts]
    out2 = model.generate(
        texts2, images,
        max_new_tokens=3,
        temperature=0.0,   # greedy
        top_k=1,           # pick argmax
        stop_on_eos=True,  # should stop immediately, append nothing
    )
    end_lens2 = [t.numel() for t in out2]
    for i in range(B):
        assert end_lens2[i] == start_lens[i], f"sample {i}: should stop on EOS immediately"
    print("Test 2 (EOS stop) OK.")

    # ---------- Test 3: all-empty-images smoke ----------
    texts3 = [torch.randint(0, 100, (4,), device=dev) for _ in range(2)]
    images3 = [[], []]  # everyone has no images
    out3 = model.generate(texts3, images3, max_new_tokens=2)
    assert all(t.numel() >= 4 for t in out3), "no-images case failed to generate"
    print("Test 3 (no images) OK.")

    print("All generate() smoke tests passed.")

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", "cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerVLM(img_size=224, patch_size=16, n_embd=128, n_layer=2, n_head=4)
    model.to(device)

    print('-_' * 50)
    print("Model parameters:", count_parameters(model))
    print('-_' * 50)

    batch_size = 4
    txt_context_size = [10, 15, 8, 200]

    # Dummy token IDs (text)
    text_tokens = [torch.randint(0, 16000, (txt_context_size[i],), device=device) for i in range(batch_size)]

    # Dummy images: each batch entry has different numbers of images
    images = [
        [torch.randn(3, 224, 224) for _ in range(2)],  # first example: 2 images
        [torch.randn(3, 224, 224)],                     # second example: 1 image
        [torch.randn(3, 224, 224) for _ in range(3)],  # third example: 3 images
        [],
    ]

    # Dummy targets (same size as text_tokens for loss computation)
    # targets = torch.randint(0, 16000, (batch_size, txt_context_size), device=device)

    logits, loss = model(text_tokens, images, calculate_loss=True)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item() if loss is not None else None)

    print("!" * 100)

    run_generate_smoke_tests(device)
