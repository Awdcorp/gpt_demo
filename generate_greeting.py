import torch
from gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# ✅ Load tokenizer
tokenizer = GPT2TokenizerWrapper()

# ✅ Load model with same config
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

# ✅ Text generation (greedy decoding or top-k sampling)
@torch.no_grad()
def generate(prompt, max_new_tokens=20, top_k=0):
    input_ids = tokenizer.encode(prompt)
    input_ids = input_ids[:max_seq_len]  # truncate if needed
    input_ids = torch.tensor([input_ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        if input_ids.shape[1] >= max_seq_len:
            break

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :]  # (batch, vocab)
        
        if top_k > 0:
            values, indices = torch.topk(next_token_logits, top_k)
            probs = torch.nn.functional.softmax(values, dim=-1)
            next_token = indices[0, torch.multinomial(probs, num_samples=1)]
        else:
            next_token = torch.argmax(next_token_logits, dim=-1)[0]

        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

    return tokenizer.decode(input_ids[0].tolist())

# ✅ Run interactive session
if __name__ == "__main__":
    print("🧠 MiniGPT Greeting Generator")
    print("------------------------------")
    while True:
        prompt = input("Enter a greeting or question (or type 'exit'): ").strip()
        if prompt.lower() == "exit":
            break

        output = generate(prompt, max_new_tokens=10)
        print(f"\n🗨️  MiniGPT: {output}\n")
