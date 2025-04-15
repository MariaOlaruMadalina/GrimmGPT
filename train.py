import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import math

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = 'cuda' 
eval_iters = 200 # câte batch-uri să folosească pentru evaluare
batch_size = 32 # Dacă ai suficientă memorie GPU; dacă nu, rămâi la 4
block_size = 128 # Lungime mai mare ajută la context mai bogat
max_iters = 5000 # Începe cu mai puține iterații ca să vezi rapid tendințele
eval_interval = 250 # Evaluează mai des pentru feedback rapid
learning_rate = 3e-4  # Mai mare la început pentru a învăța rapid
n_embd = 64 # Mai mic pentru stabilitate la început
n_head = 8 # Păstrează numărul mic pentru complexitate redusă
n_layer = 6 # Bun pentru început; dacă vezi că modelul stagnează, poți crește la 3-4
dropout = 0.4 # Mai mic pentru început ca să nu limitezi capacitatea de învățare
warmup_iters = 500 # Reduce timpul de warmup ca să înceapă învățarea mai repede
total_iters = max_iters # Menține antrenarea până la capăt


torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenizer.get_vocab_size()
print(f"Vocabular tokenizer: {vocab_size} token-uri")

# Funcții de encode și decode
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

# Încarcă datele
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Tokenizează întregul text
data = torch.tensor(encode(text), dtype=torch.long)

# Împarte datele în seturi de antrenare și validare
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    #genereaza un pachet mic de date de input x si tinte y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out={}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        v = self.value(x) # (B,T,hs)

        # Flash Attention + masca pentru autoregresie
        out = F.scaled_dot_product_attention(q, k, v, attn_mask = self.tril[:x.size(1), :x.size(1)].bool(), dropout_p=dropout)
        
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parellel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.LayerNorm(4 * n_embd),  # Folosește LayerNorm în loc de BatchNorm1d
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#model de limbaj Bigram prezice următorul cuvânt într-o propoziție pe baza cuvântului anterior
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        #Fiecare cuvânt va fi reprezentat ca un vector (un număr de valori), iar acest vector va fi folosit pentru a face predicții
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    #idx sunt cuvintele de intrare (adică ceea ce modelul folosește pentru a face predicția), 
    # iar targets sunt țintele sau cuvintele pe care modelul trebuie să le prezică (acestea sunt valorile adevărate)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if idx.max() >= vocab_size:
            raise ValueError(f"idx conține valori invalide (max={idx.max()}, vocab_size={vocab_size})")
        T = min(T, block_size)  # Limitează T la block_size
        #logits sunt reprezentările vectoriale ale cuvintelor de intrare
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))

        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        #Prelucram datele pentru a indeplini cerintele pytorch pentru cross entropy

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #Cross-entropy loss măsoară diferența între predicțiile modelului și valorile reale (țintele). 
            # Este o metodă folosită pentru probleme de clasificare, unde modelul trebuie să prezică care 
            # este cel mai probabil element dintr-o mulțime de posibilități (în cazul nostru, din vocabularul de cuvinte).
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=10):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]

            # Obține predicțiile
            logits, _ = self(idx_cond)  # logits = (B, T, vocab_size)
            logits = logits[:, -1, :]  # Focus pe ultimul token (B, vocab_size)

            # ✅ Temperatură → Scalăm logits-urile în funcție de temperatură
            logits = logits / temperature

            # ✅ Top-k sampling → Selectăm doar cele mai probabile k token-uri
            if top_k is not None:
                # Sortăm logits descrescător și luăm doar primele k valori
                values, indices = torch.topk(logits, k=top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, indices, values)

            # ✅ Softmax → Convertim logits în probabilități
            probs = F.softmax(logits, dim=-1)

            # ✅ Sampling → Alegem un token pe baza distribuției de probabilități
            idx_next = torch.multinomial(probs, num_samples=1)

            # Adăugăm token-ul la secvență
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

#Creăm un obiect din clasa BigramLanguageModel, ceea ce înseamnă că am creat un model cu dimensiunea vocabularului dată
model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

#Creăm un Pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

# Încearcă să încarci checkpoint-ul dacă există
checkpoint_path = 'checkpoint.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)  # Defaults to 0 if 'epoch' is not found

    print(f"Continuing from epoch {start_epoch}")

# Definim funcția de programare a ratei de învățare
def lr_lambda(current_step):
    if current_step < warmup_iters:
        # Warmup → creștere liniară a ratei de învățare
        return float(current_step) / float(max(1, warmup_iters))
    # Cosine decay → după warmup, descreștere conform unei curbe cosinus
    progress = float(current_step - warmup_iters) / float(max(1, total_iters - warmup_iters))
    return 0.5 * (1.0 + math.cos(math.pi * progress))  # Folosește math.cos în loc de torch.cos

# Inițializăm scheduler-ul
warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # Actualizează ReduceLROnPlateau pe baza val loss
        plateau_scheduler.step(losses['val'])

    # Obține un batch de date
    xb, yb = get_batch('train')

    # Evaluează și actualizează modelul
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Actualizează LambdaLR (warmup + decay)
    warmup_scheduler.step()

    # Salvarea checkpoint-ului
    if iter % 500 == 0:  # De exemplu, la fiecare 500 de iterații
        torch.save({
            'epoch': iter,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=500, temperature=0.8, top_k=30)
print(decode(generated[0].tolist()))
