import torch
import json
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

# === Prompt to trace ===
prompt = "hello world"
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load model ===
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)
model.eval()
model.to(device)

# === Load tokenizer ===
tokenizer = GPTTokenizer(tokenizer_path)
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)

# === Forward pass with trace ===
with torch.no_grad():
    logits, token_vectors, _ = model(input_tensor, return_vectors=True)

# === Decode output token (prediction)
predicted_id = torch.argmax(logits[0, -1]).item()
predicted_token = tokenizer.decode([predicted_id])

# === Prepare trace output ===
trace_data = {
    "input_prompt": prompt,
    "input_tokens": tokenizer.decode(input_ids).split(),
    "token_ids": input_ids,
    "token_count": len(input_ids),
    "embedding": token_vectors["embedding"].cpu().tolist(),     # (1, T, D)
    "after_attention": [layer[0].cpu().tolist() for layer in token_vectors["after_attn"]],  # list of (1, T, D)
    "after_ffn": [layer[0].cpu().tolist() for layer in token_vectors["after_ffn"]],         # list of (1, T, D)
    "predicted_token": predicted_token,
    "predicted_token_id": predicted_id
}

# === Save to JSON ===
import os
os.makedirs("visualizations_3d/assets", exist_ok=True)
with open("visualizations_3d/assets/trace.json", "w") as f:
    json.dump(trace_data, f, indent=2)

print(f"✅ Trace exported for prompt: \"{prompt}\"")
print("📄 Saved to visualizations_3d/assets/trace.json")
