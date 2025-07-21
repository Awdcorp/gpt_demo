# generate.py

import torch
import torch.nn.functional as F
from gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# Load tokenizer
tokenizer = GPT2TokenizerWrapper()

# Initialize model
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 🔁 Token selection with top-k and repetition penalty
def sample_next_token(logits, past_tokens, temperature=1.0, top_k=5, repetition_penalty=1.2):
    logits = logits / temperature

    # Apply repetition penalty
    if past_tokens:
        for token_id in past_tokens:
            logits[0, token_id] /= repetition_penalty

    # Top-k filtering
    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return indices[0, next_token.item()]
    else:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()


# 🔁 Sampling loop
@torch.no_grad()
def generate(prompt, max_new_tokens=20, temperature=1.0, top_k=5, repetition_penalty=1.2):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids], dtype=torch.long)  # (1, seq_len)
    past_tokens = input_ids[0].tolist()

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        # Forward pass
        logits, _ = model(input_ids)

        # Get logits of last token
        next_token_logits = logits[:, -1, :]  # (1, vocab_size)

        # Sample next token using controlled sampling
        next_token_id = sample_next_token(next_token_logits, past_tokens, temperature, top_k, repetition_penalty)

        # Append to sequence
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)
        past_tokens.append(next_token_id)

    return tokenizer.decode(input_ids[0].tolist())


# ▶️ Example usage
if __name__ == "__main__":
    prompt = "How are you"
    output = generate(
        prompt,
        max_new_tokens=30,
        temperature=1.0,
        top_k=5,
        repetition_penalty=1.2
    )
    print("Generated:\n", output)
