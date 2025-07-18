# generate.py

import torch
import torch.nn.functional as F
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (  # ✅ Import shared settings
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# Load tokenizer
tokenizer = GPTTokenizer(tokenizer_path)

# Initialize model and load checkpoint
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))  # ✅ Load trained weights
model.eval()


# 🔁 Token selection: greedy / top-k / top-p
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        topk_probs = topk_vals / torch.sum(topk_vals)
        next_token = topk_idx[torch.multinomial(topk_probs, 1)]
        return next_token.item()

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= torch.sum(sorted_probs)
        next_token = sorted_idx[torch.multinomial(sorted_probs, 1)]
        return next_token.item()

    return torch.argmax(probs).item()


# 🔁 Sampling loop
@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids])  # shape: (1, seq_len)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # Choose next token
        next_token_id = sample_next_token(next_token_logits[0], temperature, top_k, top_p)

        # Append token
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# 🔁 Interactive CLI
if __name__ == "__main__":
    print("🧠 MiniGPT Text Generator")
    print("-" * 30)

    prompt = input("Enter your prompt: ").strip()

    max_new_tokens = input("How many tokens to generate? [default=30]: ").strip()
    max_new_tokens = int(max_new_tokens) if max_new_tokens else 30

    top_k = input("Top-k sampling (enter number or leave blank): ").strip()
    top_k = int(top_k) if top_k else None

    top_p = input("Top-p (nucleus) sampling (0.0–1.0, leave blank if unused): ").strip()
    top_p = float(top_p) if top_p else None

    temperature = input("Temperature? [default=1.0]: ").strip()
    temperature = float(temperature) if temperature else 1.0

    print("\n⏳ Generating...\n")
    output = generate(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)

    print("📝 Generated:\n")
    print(output)