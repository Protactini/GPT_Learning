import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken #Wrod tokennizer from OpenAI

# Setting tokeniser to a specific model  
enc = tiktoken.get_encoding("cl100k_base")

# Pparameters
batch_size = 16 # how many independent sequences will be process in parallel
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000 # how many time does the block of learning will be repeated 
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Using tiktoken to encode test and splite into train and test 
data = torch.tensor(enc.encode(text), dtype=torch.long)
vocab_size = max(data); # Get size for the ml network
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()

# Adding the most out layer of the NN
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token should directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        #### here should be a block of nural network which is the main part of this code####
        
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    #Where the data will be trained
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    # Generate result of this nn
    def generate(self, idx, max_new_tokens):
        # 


model = BigramLanguageModel()
m = model.to(device)