# train.py

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (  # ✅ Import all from config.py
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    batch_size, learning_rate, epochs,
    tokenizer_path, corpus_path, checkpoint_path
)
import os

# 1. Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Load tokenizer and raw text
tokenizer = GPTTokenizer(tokenizer_path)
with open(corpus_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

# 3. Tokenize and chunk into fixed-length sequences
tokens = tokenizer.encode(raw_text)
sequences = []
for i in range(0, len(tokens) - max_seq_len, max_seq_len):
    chunk = tokens[i : i + max_seq_len]
    sequences.append(chunk)

print(f"✅ Total training sequences: {len(sequences)}")

# 4. Create Dataset and DataLoader
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.data = torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        return x, y  # input_ids, labels

dataset = TextDataset(sequences)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 5. Load model and optimizer
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# 6. Training Loop
model.train()
for epoch in range(epochs):
    total_loss = 0.0

    for batch in dataloader:
        input_ids, labels = [b.to(device) for b in batch]
        
        logits, loss = model(input_ids, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"🧠 Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

# 7. Save model
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
torch.save(model.state_dict(), checkpoint_path)
print(f"✅ Model saved to {checkpoint_path}")
