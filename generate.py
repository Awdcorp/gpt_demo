# generate.py

import torch
import torch.nn.functional as F
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (  # ✅ Import shared settings
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# pick device and keep model/tensors consistent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer
tokenizer = GPTTokenizer(tokenizer_path)

# optional stop tokens if your tokenizer has them
try:
    EOS_ID = tokenizer.encode("[EOS]")[0]
except Exception:
    EOS_ID = None
try:
    EOT_ID = tokenizer.encode("[EOT]")[0]
except Exception:
    EOT_ID = None

# Initialize model and load checkpoint
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(DEVICE)
model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))  # ✅ Load trained weights
model.eval()


# 🔁 Token selection: greedy / top-k / top-p
def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    logits: 1D tensor of shape [vocab_size]
    Applies temperature, optional top-k/top-p on LOGITS, then samples.
    """
    # guard temperature
    if temperature <= 0:
        # effectively greedy
        return int(torch.argmax(logits))

    # scale by temperature
    logits = logits / temperature

    # apply top-k on logits first (standard practice)
    if top_k is not None and top_k > 0 and top_k < logits.numel():
        topk_vals, topk_idx = torch.topk(logits, top_k)
        filtered = torch.full_like(logits, float("-inf"))
        filtered[topk_idx] = topk_vals
        logits = filtered

    # apply top-p (nucleus) on logits
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        # mask tokens beyond nucleus
        cutoff = cumulative > top_p
        # keep the first token above the threshold as well
        cutoff[..., 1:] = cutoff[..., :-1].clone()
        cutoff[..., 0] = False
        sorted_logits[cutoff] = float("-inf")
        # unsort back to original indices
        logits = torch.full_like(logits, float("-inf"))
        logits[sorted_idx] = sorted_logits

    # final sample
    probs = F.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, 1)
    return int(next_token.item())


# 🔁 Sampling loop
@torch.no_grad()
def generate(prompt, max_new_tokens=30, temperature=1.0, top_k=None, top_p=None):
    #keep a python list of ids; feed a sliding window to model
    input_ids = tokenizer.encode(prompt)
    generated_ids = list(input_ids)

    for _ in range(max_new_tokens):
        # create the model input as the last max_seq_len tokens
        ctx = generated_ids[-max_seq_len:] if len(generated_ids) > max_seq_len else generated_ids
        input_tensor = torch.tensor([ctx], device=DEVICE, dtype=torch.long)

        # forward
        logits, _, _, _ = model(input_tensor)
        next_token_logits = logits[:, -1, :].squeeze(0)  # (vocab_size,)

        # choose next token
        next_token_id = sample_next_token(next_token_logits, temperature, top_k, top_p)

        # append
        generated_ids.append(next_token_id)

        # stop when EOS/EOT appears
        if (EOS_ID is not None and next_token_id == EOS_ID) or (EOT_ID is not None and next_token_id == EOT_ID):
            break

    return tokenizer.decode(generated_ids)


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