# test_pipeline.py

import torch
from tokenizer.tokenizer import GPTTokenizer
from model.embedding import GPTEmbedding
from model.block import TransformerBlock
from model.attention import MultiHeadSelfAttention
from model.gpt_model import GPTBackbone, MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    tokenizer_path, checkpoint_path
)

# 1. Load tokenizer and encode input
tokenizer = GPTTokenizer(tokenizer_path)
text = "The quick brown fox"
token_ids = tokenizer.encode(text)
print("Token IDs:", token_ids)

# 2. Convert token IDs to tensor
input_ids = torch.tensor([token_ids])  # shape: (1, seq_len)

# 3. Initialize modules
embedding_layer = GPTEmbedding(vocab_size, embed_dim, max_seq_len)
attention_layer = MultiHeadSelfAttention(embed_dim, num_heads)
block = TransformerBlock(embed_dim, num_heads, ff_dim)
gpt = GPTBackbone(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)
minigpt = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers)

# Optional: Load trained weights (comment if testing untrained)
try:
    minigpt.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    print("✅ Loaded trained weights for MiniGPT.")
except FileNotFoundError:
    print("⚠️ Trained model not found. Running with randomly initialized weights.")

# 4. Test embedding
embeddings = embedding_layer(input_ids)
print("\nEmbedding output shape:", embeddings.shape)  # (1, seq_len, embed_dim)

# 5. Test attention
attn_out = attention_layer(embeddings)
print("Attention output shape:", attn_out.shape)  # (1, seq_len, embed_dim)

# 6. Test transformer block
block_out = block(embeddings)
print("TransformerBlock output shape:", block_out.shape)

# 7. Test GPTBackbone
gpt_out = gpt(input_ids)
print("GPTBackbone output shape:", gpt_out.shape)

# 8. Test MiniGPT forward only
logits, loss = minigpt(input_ids)
print("MiniGPT Logits shape:", logits.shape)
print("Loss (none expected):", loss)

# 9. Test MiniGPT with labels
logits, loss = minigpt(input_ids, labels=input_ids)
print("Loss with labels:", loss.item())
