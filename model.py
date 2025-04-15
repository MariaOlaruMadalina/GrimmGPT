import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu' # GPU sau CPU
eval_iters = 200 # câte batch-uri să folosească pentru evaluare

batch_size = 32 # Dacă ai suficientă memorie GPU; dacă nu, rămâi la 4
block_size = 128 # Lungime mai mare ajută la context mai bogat
max_iters = 5000 # Începe cu mai puține iterații ca să vezi rapid tendințele
eval_interval = 250 # Evaluează mai des pentru feedback rapid
learning_rate = 1e-4 # Mai mare la început pentru a învăța rapid
n_embd = 64 # Mai mic pentru stabilitate la început
n_head = 8 # Păstrează numărul mic pentru complexitate redusă
n_layer = 6 # Bun pentru început; dacă vezi că modelul stagnează, poți crește la 3-4
dropout = 0.4 # Mai mic pentru început ca să nu limitezi capacitatea de învățare
warmup_iters = 500 # Reduce timpul de warmup ca să înceapă învățarea mai repede
total_iters = max_iters # Menține antrenarea până la capăt

tokenizer = Tokenizer.from_file("tokenizer.json")

vocab_size = tokenizer.get_vocab_size()

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