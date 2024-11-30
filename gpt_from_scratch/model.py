import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparams
batch_size = 4
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 5e-4
# device = 'cpu'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 200
n_embed = 32
n_head = 4
n_layer = 4
dropout = 0.1

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Unique characters that occur in the text
# Creating vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
# Create mapping from int to char and vice versa
stoi = { ch: i for i, ch in enumerate(chars)}
itos = { i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(s):
    return ''.join([itos[i] for i in s])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
  
class Head(nn.Module):
    """ One head of self attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # Compute attention scores 'affinities'
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        # Perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple heads of self-attention in parallel """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList((Head(head_size) for _ in range(num_heads)))
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Concatenate the outputs over the channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple linear layer followed by non-linearity to allow nodes to 'think' about the information they gained """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed), # Inner layer has dimensionality 4x that of input and output
            nn.GELU(), # ATTENTION IS ALL YOU NEED uses a ReLU, GPT uses a GELU
            nn.Linear(4 * n_embed, n_embed), # Projection layer
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: Communication followed by computation """
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Adding residual connections (the addition - fork off and do some computation)
        x = x + self.ff(self.ln2(x))
        return x

# Bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Each token directly reads off the logits for the next token in a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed) # Each position gets its own embedding as well
        # self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 heads of 8-dimensional self attention, get dimension of 32
        # self.ff = FeedForward(n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embed)
       
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and target are both (B, T) tensors of integers
        token_emb = self.token_embedding_table(idx) # B, T, C (4, 8, vocab_size)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = token_emb + pos_emb
        # x = self.sa_heads(x) # Apply one head of self attention. (B, T, C)
        # x = self.ff(x)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        # Batch, by time, by channel tensor

        # Note the way that the input needs to be formatted for the functional version of cross entropy
        # Look at documentation
        # Pytorch wants B * T, C instead
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets) 
            return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # Get predictinons
            logits = self(idx_cond)
            # Focus only on the last step
            # Does not look at full history
            logits, loss = logits[:, -1, :] # Becomes B, C
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx
  
model = BigramLanguageModel()
m = model.to(device)

# Create optimiser
optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

for iter in range(max_iters):
   
   # Every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    xb, yb = get_batch('train') 

    # Evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=False)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# This is essentially the pre-training, decoder only side which just allows the model to babble on with text and create words (from just characters at the moment)