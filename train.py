import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    batch_size, learning_rate, epochs,
    tokenizer_path, corpus_path, checkpoint_path
)
import os
import time

# ✅ Optional debug mode
TEST_MODE = False
MAX_BATCHES = 300

# ✅ Save model every N batches
SAVE_EVERY = 1000

# ✅ Dataset class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.data = torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = x.clone()
        return x, y

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # ✅ Load tokenizer and raw text
    tokenizer = GPTTokenizer(tokenizer_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ✅ Tokenize and split
    tokens = tokenizer.encode(raw_text)
    sequences = [tokens[i:i + max_seq_len] for i in range(0, len(tokens) - max_seq_len, max_seq_len)]
    if TEST_MODE:
        sequences = sequences[:10000]
    print(f"✅ Total training sequences: {len(sequences)}")

    # ✅ DataLoader setup
    dataset = TextDataset(sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),  # Use all CPU cores
        pin_memory=torch.cuda.is_available()  # Only pin memory if CUDA
    )

    # ✅ Model + optimizer
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ✅ Resume from checkpoint (optional)
    if os.path.exists(checkpoint_path):
        print(f"📦 Resuming from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path))

    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ✅ Mixed Precision for GPU
    use_amp = torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        start = time.time()

        for batch_idx, batch in enumerate(dataloader):
            if TEST_MODE and batch_idx >= MAX_BATCHES:
                break

            input_ids, labels = [b.to(device) for b in batch]

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    logits, loss = model(input_ids, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss = model(input_ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # 💬 Logging
            if batch_idx % 100 == 0:
                print(f"  [Epoch {epoch+1}] Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

            # 💾 Save intermediate checkpoint
            if (batch_idx + 1) % SAVE_EVERY == 0:
                partial_path = checkpoint_path.replace(".pt", f"_e{epoch+1}_b{batch_idx+1}.pt")
                os.makedirs(os.path.dirname(partial_path), exist_ok=True)
                torch.save(model.state_dict(), partial_path)
                print(f"💾 Saved intermediate model at {partial_path}")

        avg_loss = total_loss / (batch_idx + 1)
        print(f"🧠 Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f} | Time: {time.time() - start:.1f}s")

        # ✅ Final save after each epoch
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model saved to {checkpoint_path}")

# ✅ For multiprocessing on Windows
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
