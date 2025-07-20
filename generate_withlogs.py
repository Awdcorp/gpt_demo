import torch
import torch.nn.functional as F
import numpy as np
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

import datetime
import os

# Prepare log file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"generation_{timestamp}.log"
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# Load tokenizer
tokenizer = GPTTokenizer(tokenizer_path)

# Initialize model and load checkpoint
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# 🧩 Print top-k logits (before softmax)
def print_top_k_logits(logits, k=5):
    topk_vals, topk_idx = torch.topk(logits, k)
    log("📊 Raw Logits (Top-5):")
    for i in range(k):
        tid = topk_idx[i].item()
        val = topk_vals[i].item()
        decoded = tokenizer.decode([tid])
        log(f"  {i+1}. Token ID: {tid} → '{decoded}' | Logit: {val:.4f}")

# 🔍 Print top-k softmax tokens
def print_top_k_tokens(logits, k=5):
    probs = F.softmax(logits, dim=-1)
    topk_vals, topk_idx = torch.topk(probs, k)
    log("🔍 Top-k Softmax Probabilities:")
    for i in range(k):
        tid = topk_idx[i].item()
        token = tokenizer.decode([tid])
        prob = topk_vals[i].item()
        log(f"  {i+1}. Token ID: {tid} → '{token}' (prob={prob:.4f})")

# 🎨 Render attention matrix as text heatmap
def render_attention_heatmap(attn_weights, tokens):
    log("🧲 Attention Heatmap [Head 0]:")
    attn = attn_weights[0, 0]  # [seq_len, seq_len]
    attn = attn.detach().cpu().numpy()
    tokens = [t.replace('\n', ' ') for t in tokens]
    header = "     " + " ".join([f"{t:>5}" for t in tokens])
    log(header)
    for i, row in enumerate(attn):
        row_vals = " ".join([f"{w:.2f}" for w in row])
        log(f"{tokens[i]:>4} {row_vals}")

# 🔁 Token selection logic
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    if top_k:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        topk_probs = topk_vals / torch.sum(topk_vals)
        next_token = topk_idx[torch.multinomial(topk_probs, 1)]
        return next_token.item()

    if top_p:
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

# 🔁 Generation loop with all logs
@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None):
    input_ids = tokenizer.encode(prompt)
    log(f"\n📥 Input Prompt: {prompt}")
    log(f"🔢 Tokenized Input IDs: {input_ids}")
    log(f"🧩 Tokens: {[tokenizer.decode([i]) for i in input_ids]}\n")

    input_ids = torch.tensor([input_ids])
    generated_ids = input_ids[0].tolist()

    for step in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :]

        # 🧬 Embedding vector (last token)
        if hasattr(model.backbone, 'embedding'):
            embed = model.backbone.embedding(input_ids)
            log("🧬 Embedding Vector (last token):")
            log(str(embed[0, -1, :].detach().cpu().numpy()))

        # 📊 Raw logits
        print_top_k_logits(next_token_logits[0], k=5)

        # 🔍 Softmax token sampling
        print_top_k_tokens(next_token_logits[0], k=5)

        # 🧠 Next token chosen
        next_token_id = sample_next_token(next_token_logits[0], temperature, top_k, top_p)
        token_str = tokenizer.decode([next_token_id])
        log(f"[{step+1:02d}] 🧠 Chosen Token ID: {next_token_id} → '{token_str}'")

        # 🧲 Attention heatmap
        if hasattr(model.backbone.blocks[-1].attn, 'attn_weights'):
            tokens_so_far = [tokenizer.decode([i]) for i in generated_ids]
            render_attention_heatmap(model.backbone.blocks[-1].attn.attn_weights, tokens_so_far)

        log("—" * 40)

        generated_ids.append(next_token_id)
        input_ids = torch.tensor([generated_ids])

    final_output = tokenizer.decode(generated_ids)
    log("\n📤 Final Generated Token IDs:")
    log(str(generated_ids))
    log("🧾 Final Output Text:")
    log(final_output)

    # 📝 Save full log
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\n📝 Full log saved to: {log_file_path}")

    return final_output

# 🔁 CLI Runner
if __name__ == "__main__":
    print("🧠 MiniGPT Generator — Explainable Mode")
    print("-" * 30)

    prompt = input("Enter your prompt: ").strip()
    max_new_tokens = input("How many tokens to generate? [default=30]: ").strip()
    max_new_tokens = int(max_new_tokens) if max_new_tokens else 30

    top_k = input("Top-k sampling (enter number or leave blank): ").strip()
    top_k = int(top_k) if top_k else None

    top_p = input("Top-p (nucleus) sampling, 0.0–1.0 (leave blank to disable): ").strip()
    top_p = float(top_p) if top_p else None

    temperature = input("Temperature? [default=1.0]: ").strip()
    temperature = float(temperature) if temperature else 1.0

    print("\n⏳ Generating with full trace...\n")
    output = generate(prompt, max_new_tokens, temperature, top_k, top_p)

    print("\n📝 Final Generated Text:\n")
    print(output)
