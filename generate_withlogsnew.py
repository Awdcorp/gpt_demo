# generate_withlogs.py

import torch
import json
import os
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

# === Forward pass with token vectors + attention ===
with torch.no_grad():
    logits, token_vectors, attn_weights = model(input_tensor, return_vectors=True, return_attn=True)

# === Decode predicted token
predicted_id = torch.argmax(logits[0, -1]).item()
predicted_token = tokenizer.decode([predicted_id])

# === Convert attention weights to JSON-friendly format
attn_json = {}
if attn_weights:
    for layer_idx, layer_heads in enumerate(attn_weights):
        attn_json[f"layer_{layer_idx}"] = {}
        for head_idx in range(layer_heads.shape[1]):  # shape: (B, H, T, T)
            attn_map = layer_heads[0, head_idx].cpu().tolist()  # (T, T)
            attn_json[f"layer_{layer_idx}"][f"head_{head_idx}"] = attn_map

# === Create trace output
trace_data = {
    "input_prompt": prompt,
    "input_tokens": tokenizer.decode(input_ids).split(),
    "token_ids": input_ids,
    "predicted_token": predicted_token,
    "token_vectors": {
        "embedding": token_vectors["embedding"][0].tolist(),  # (T, D)
        "after_attn": [layer[0].tolist() for layer in token_vectors["after_attn"]],  # (L, T, D)
        "after_ffn": [layer[0].tolist() for layer in token_vectors["after_ffn"]]     # (L, T, D)
    },
    "attn_weights": attn_json
}

# === Save to trace.json
output_path = "visualizations_3d/assets/trace.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(trace_data, f, indent=2)

print(f"✅ Trace saved to {output_path}")
