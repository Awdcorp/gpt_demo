import torch
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

# === Load tokenizer ===
tokenizer = GPTTokenizer(tokenizer_path)

# === Load model & checkpoint ===
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# === Test prompt ===
prompt = "hello"
ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([ids])

# === Forward pass ===
with torch.no_grad():
    logits, _, _, _ = model(input_tensor)
    next_token_logits = logits[:, -1, :]  # prediction for next token

# === Top-5 predictions ===
topk_vals, topk_idx = torch.topk(next_token_logits, k=5, dim=-1)
print(f"\nPrompt: {prompt}")
print("Top-5 predicted next tokens:\n")
for rank, (tid, val) in enumerate(zip(topk_idx[0], topk_vals[0]), start=1):
    token_str = tokenizer.decode([tid.item()])
    print(f"{rank}. Token: '{token_str}' | Logit: {val.item():.4f}")