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
from collections import Counter

# Prepare log file
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_path = f"generation_{timestamp}.log"
log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

# Load tokenizer
tokenizer = GPTTokenizer(tokenizer_path)
#tokenizer = GPTTokenizer(tokenizer_path, trace=True, logger=log)

#  EOT for real chat-style stopping (safe even if unused)
try:
    EOT_ID = tokenizer.encode("[EOT]")[0]
except Exception:
    EOT_ID = None

#  EOS for standard end-of-sentence stopping
try:
    EOS_ID = tokenizer.encode("[EOS]")[0]
except Exception:
    EOS_ID = None

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

# show probability of the chosen token
def log_chosen_token_prob(logits, token_id):
    probs = F.softmax(logits, dim=-1)
    prob = probs[token_id].item()
    log(f"   ↳ Probability of chosen token: {prob:.4f}")

# >>> NEW TRACE — attention for all heads
def render_attention_heatmap(attn_weights, tokens):
    layer_attn = attn_weights[0]  # [heads, seq_len, seq_len]
    tokens = [t.replace("\n", " ") for t in tokens]
    for head_idx in range(layer_attn.size(0)):
        log(f"🧲 Attention Heatmap — Head {head_idx}:")
        attn = layer_attn[head_idx].detach().cpu().numpy()
        header = "     " + " ".join([f"{t:>5}" for t in tokens])
        log(header)
        for i, row in enumerate(attn):
            row_vals = " ".join([f"{w:.2f}" for w in row])
            log(f"{tokens[i]:>4} {row_vals}")

# >>> NEW TRACE — log embeddings for all tokens
def log_all_embeddings(input_ids):
    embed = model.backbone.embedding(input_ids)[0].detach().cpu().numpy()
    log("🧬 Embedding Vectors (all tokens so far):")
    for idx, vec in enumerate(embed):
        token_str = tokenizer.decode([input_ids[0, idx].item()])
        log(f"  Token {idx} '{token_str}': {vec}")

# 🔁 Token selection logic
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None, token_ids_so_far=None, repetition_penalty=1.0):
    # Apply repetition penalty
    if repetition_penalty != 1.0 and token_ids_so_far:
        token_counts = Counter(token_ids_so_far)
        for token_id, count in token_counts.items():
            logits[token_id] /= (repetition_penalty ** count)

    # Apply temperature
    logits = logits / temperature

    # Apply top-p sampling
    if top_p is not None:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        logits[sorted_indices[sorted_indices_to_remove]] = -float("Inf")

    # Apply top-k sampling
    if top_k is not None:
        topk_vals, topk_idx = torch.topk(logits, top_k)
        logits = torch.full_like(logits, float('-inf'))
        logits[topk_idx] = topk_vals

    # Final token selection
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()

# 🔁 Generation loop with full logs
@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0):
    input_ids = tokenizer.encode(prompt)
    log(f"\n📥 Input Prompt: {prompt}")
    log(f"🔢 Tokenized Input IDs: {input_ids}")
    log(f"🧩 Tokens: {[tokenizer.decode([i]) for i in input_ids]}\n")

    input_ids = torch.tensor([input_ids])
    generated_ids = input_ids[0].tolist()

    for step in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        # get attention weights properly
        logits, _, attn_layers, _ = model(input_ids, return_attn=True)
        next_token_logits = logits[:, -1, :]

        # 🧬 Embedding vector (last token)
        log_all_embeddings(input_ids)

        # 📊 Raw logits
        print_top_k_logits(next_token_logits[0], k=5)

        # 🔍 Softmax tokens
        print_top_k_tokens(next_token_logits[0], k=5)

        # 🧠 Sample next token
        next_token_id = sample_next_token(
            next_token_logits[0],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            token_ids_so_far=generated_ids,
            repetition_penalty=repetition_penalty
        )
        token_str = tokenizer.decode([next_token_id])
        log(f"[{step+1:02d}] 🧠 Chosen Token ID: {next_token_id} → '{token_str}'")
        log_chosen_token_prob(next_token_logits[0], next_token_id)

        # 🧲 Attention (prefer returned layers; keep old path as fallback)
        if attn_layers and len(attn_layers) > 0:
            tokens_so_far = [tokenizer.decode([i]) for i in generated_ids]
            render_attention_heatmap(attn_layers[-1], tokens_so_far)
        elif hasattr(model.backbone.blocks[-1].attn, 'attn_weights'):
            tokens_so_far = [tokenizer.decode([i]) for i in generated_ids]
            render_attention_heatmap(model.backbone.blocks[-1].attn.attn_weights, tokens_so_far)

        log("—" * 40)

        generated_ids.append(next_token_id)

        # optional stop by EOS or EOT
        if ((EOT_ID is not None and next_token_id == EOT_ID) or
            (EOS_ID is not None and next_token_id == EOS_ID)):
            break

        input_ids = torch.tensor([generated_ids])

    final_output = tokenizer.decode(generated_ids)
    log("\n📤 Final Generated Token IDs:")
    log(str(generated_ids))
    log("🧾 Final Output Text:")
    log(final_output)

    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"\n📝 Full log saved to: {log_file_path}")

    return final_output

# 🔁 CLI Runner
if __name__ == "__main__":
    print("🧠 MiniGPT Generator — Explainable Mode")
    print("-" * 30)

    while True:
        prompt = input("\nEnter your prompt (or 'exit' to quit): ").strip()
        if prompt.lower() in {"exit", "quit"}:
            print("👋 Exiting generator.")
            break

        max_new_tokens = input("How many tokens to generate? [default=30]: ").strip()
        max_new_tokens = int(max_new_tokens) if max_new_tokens else 30

        top_k = input("Top-k sampling (enter number or leave blank): ").strip()
        top_k = int(top_k) if top_k else None

        top_p = input("Top-p (nucleus) sampling, 0.0–1.0 (leave blank to disable): ").strip()
        top_p = float(top_p) if top_p else None

        temperature = input("Temperature? [default=1.0]: ").strip()
        temperature = float(temperature) if temperature else 1.0

        repetition_penalty = input("Repetition penalty? [default=1.0]: ").strip()
        repetition_penalty = float(repetition_penalty) if repetition_penalty else 1.0

        print("\n⏳ Generating with full trace...\n")
        output = generate(
            prompt,
            max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty
        )

        print("\n📝 Final Generated Text:\n")
        print(output)
