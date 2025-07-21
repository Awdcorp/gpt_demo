import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
#from tokenizer.tokenizer import GPTTokenizer
from gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from model.gpt_model import MiniGPT
from config import (
    vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers,
    batch_size, learning_rate, epochs,
    tokenizer_path, corpus_path, checkpoint_path
)
import os
import time

# Optional debug mode
TEST_MODE = False
MAX_BATCHES = 300
SAVE_EVERY = 1000  # More frequent saving for small dataset

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

# ✅ Training function
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # ✅ Load tokenizer and corpus
    tokenizer = GPT2TokenizerWrapper()
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ✅ Line-by-line tokenization + padding
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    sequences = []
    for line in lines:
        token_ids = tokenizer.encode(line)
        if len(token_ids) > max_seq_len:
            token_ids = token_ids[:max_seq_len]
        else:
            token_ids += [0] * (max_seq_len - len(token_ids))  # Pad
        sequences.append(token_ids)

    if TEST_MODE:
        sequences = sequences[:MAX_BATCHES]

    print(f"✅ Total training sequences: {len(sequences)}")

    # ✅ Dataset & Dataloaders (optional val split)
    dataset = TextDataset(sequences)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # ✅ Initialize model
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    if os.path.exists(checkpoint_path):
        try:
            print(f"📦 Resuming from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path))
        except RuntimeError as e:
            print(f"⚠️ Checkpoint mismatch: {e}\n🔁 Starting from scratch.")

    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ✅ Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if TEST_MODE and batch_idx >= MAX_BATCHES:
                break

            input_ids, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            if torch.cuda.is_available():
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

            # ✅ Logging and saving
            if batch_idx % 1 == 0:
                msg = f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                print("  " + msg)
                with open("training_log.txt", "a") as f:
                    f.write(msg + "\n")

            if (batch_idx + 1) % SAVE_EVERY == 0:
                partial_path = checkpoint_path.replace(".pt", f"_e{epoch+1}_b{batch_idx+1}.pt")
                os.makedirs(os.path.dirname(partial_path), exist_ok=True)
                torch.save(model.state_dict(), partial_path)
                print(f"💾 Saved checkpoint: {partial_path}")

        avg_loss = total_loss / (batch_idx + 1)
        print(f"🧠 Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_loss:.4f} | Time: {time.time() - start:.1f}s")

        # ✅ Save final model each epoch
        scheduler.step()
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Model saved to: {checkpoint_path}")

# ✅ Entry point
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
