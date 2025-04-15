import torch
from model import BigramLanguageModel  
from tokenizers import Tokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

tokenizer = Tokenizer.from_file("tokenizer.json")
encode = lambda s: tokenizer.encode(s).ids
decode = lambda l: tokenizer.decode(l)

model = BigramLanguageModel()
model.to(device)  

checkpoint_path = 'checkpoint.pth'
checkpoint = torch.load(checkpoint_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  

context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=200, temperature=0.8, top_k=30)
print(decode(generated[0].tolist()))