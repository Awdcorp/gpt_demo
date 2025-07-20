# web_ui.py

import torch
import torch.nn.functional as F
import gradio as gr
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# Load tokenizer and model
tokenizer = GPTTokenizer(tokenizer_path)
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# Token selection logic
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_k:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        topk_probs = topk_vals / torch.sum(topk_vals)
        return topk_idx[torch.multinomial(topk_probs, 1)].item()

    if top_p:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_probs[mask] = 0
        sorted_probs /= torch.sum(sorted_probs)
        return sorted_idx[torch.multinomial(sorted_probs, 1)].item()

    return torch.argmax(probs).item()

# Generation function with optional verbose logging
@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=1.0, top_k=0, top_p=0.0, show_log=False):
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids])
    generated_ids = input_ids[0].tolist()

    logs = []
    logs.append(f"📥 Prompt: {prompt}")
    logs.append(f"🔢 Tokenized: {generated_ids}")
    logs.append(f"🧩 Tokens: {[tokenizer.decode([i]) for i in generated_ids]}")

    for i in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = sample_next_token(next_token_logits[0], temperature, top_k if top_k > 0 else None, top_p if top_p > 0.0 else None)

        if show_log:
            token_str = tokenizer.decode([next_token_id])
            logs.append(f"[{i+1:02d}] → Token ID: {next_token_id} → '{token_str}'")

        generated_ids.append(next_token_id)
        input_ids = torch.tensor([generated_ids])

    final_output = tokenizer.decode(generated_ids)
    logs.append("\n📝 Final Output:\n" + final_output)

    return "\n".join(logs) if show_log else final_output

# Gradio UI
ui = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(10, 100, value=30, label="Max New Tokens"),
        gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="Temperature"),
        gr.Slider(0, 100, value=0, step=1, label="Top-k (0 = disable)"),
        gr.Slider(0.0, 1.0, value=0.0, step=0.01, label="Top-p (0.0 = disable)"),
        gr.Checkbox(label="Show step-by-step generation log")
    ],
    outputs=gr.Textbox(label="Generated Output"),
    title="🧠 MiniGPT Text Generator",
    description="Enter a prompt and adjust sampling settings to generate text from your MiniGPT model. Enable logging to see token-by-token reasoning."
)

if __name__ == "__main__":
    ui.launch()
