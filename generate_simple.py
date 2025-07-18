# generate.py

import torch
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT

# Load tokenizer and model config
tokenizer = GPTTokenizer("tokenizer/tokenizer.json")

# ---- Model Config ----
vocab_size = 1000
embed_dim = 64
max_seq_len = 128
num_heads = 4
ff_dim = 256
num_layers = 4

# Initialize model
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.eval()

# 🔁 Sampling loop
@torch.no_grad()
def generate(prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids])  # shape: (1, seq_len)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        # Get logits from the model
        logits, _ = model(input_ids)

        # Get last token's logits
        next_token_logits = logits[:, -1, :]  # shape: (1, vocab_size)

        # Greedy: pick token with highest score
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # (1,1)

        # Append to input
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return tokenizer.decode(input_ids[0].tolist())


# Example usage
if __name__ == "__main__":
    prompt = "The meaning of life"
    output = generate(prompt, max_new_tokens=30)
    print("Generated:\n", output)
