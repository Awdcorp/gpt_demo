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

# Optional debug mode
TEST_MODE = False
MAX_BATCHES = 300
SAVE_EVERY = 300  # More frequent saving for small dataset

# ✅ Dataset class
class TextDataset(Dataset):
    def __init__(self, sequences):
        self.data = torch.tensor(sequences, dtype=torch.long)

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx][:-1]      # all tokens except last → model input
        y = self.data[idx][1:]       # all tokens except first → prediction target
        return x, y

# ✅ Training function
def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    # ✅ Load tokenizer and corpus
    tokenizer = GPTTokenizer(tokenizer_path)
    with open(corpus_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # compute pad id once
    pad_token_id = tokenizer.encode("[PAD]")[0]

    # ✅ Line-by-line tokenization + padding
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    sequences = []
    for line in lines:
        # tokenization
        token_ids = tokenizer.encode(line)

        target_seq_len = max_seq_len + 1  # to allow shifting
        if len(token_ids) > target_seq_len:
            token_ids = token_ids[:target_seq_len]
        else:
            token_ids += [pad_token_id] * (target_seq_len - len(token_ids))  # Pad
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
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())  # >>> FIX: safe AMP init

    # ensure checkpoint dir exists
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    if os.path.exists(checkpoint_path):
        try:
            print(f"📦 Resuming from {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except RuntimeError as e:
            print(f"⚠️ Checkpoint mismatch: {e}\n🔁 Starting from scratch.")

    print(f"📊 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ✅ Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if TEST_MODE and batch_idx >= MAX_BATCHES:
                break

            input_ids, labels = [b.to(device) for b in batch]

            # mask PAD in labels so loss ignores them
            labels = labels.clone()
            labels[labels == pad_token_id] = -100

            optimizer.zero_grad(set_to_none=True)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    logits, loss, _, _ = model(input_ids, labels)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, loss, _, _ = model(input_ids, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()

            # ✅ Logging and saving
            if batch_idx % 1 == 0:
                msg = f"[Epoch {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}"
                print("  " + msg)
                with open("training_log.txt", "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

            # ✅ Save checkpoint and embedding snapshot every N batches
            if (batch_idx + 1) % SAVE_EVERY == 0:
                # Save model checkpoint
                partial_path = checkpoint_path.replace(".pt", f"_e{epoch+1}_b{batch_idx+1}.pt")
                os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), partial_path)
                print(f"💾 Saved checkpoint: {partial_path}")

                # Save embedding snapshot
                embedding_matrix = model.backbone.embedding.token_embedding.weight.detach().cpu()
                embed_path = f"checkpoints/embeddings_e{epoch+1}_b{batch_idx+1}.pt"
                os.makedirs(os.path.dirname(embed_path) or ".", exist_ok=True)
                torch.save(embedding_matrix, embed_path)
                print(f"📊 Saved embedding snapshot: {embed_path}")

        avg_loss = total_loss / (batch_idx + 1)
        print(f"🧠 Epoch {epoch+1}/{epochs} | Avg Train Loss: {avg_loss:.4f} | Time: {time.time() - start:.1f}s")

        # ✅ Save final model checkpoint and embedding at end of epoch
        scheduler.step()
        torch.save(model.state_dict(), checkpoint_path)
        model.eval()
        print(f"✅ Final model saved to: {checkpoint_path}")

        embedding_matrix = model.backbone.embedding.token_embedding.weight.detach().cpu()
        embed_path = f"checkpoints/embeddings_e{epoch+1}_final.pt"
        torch.save(embedding_matrix, embed_path)
        print(f"📊 Final embedding snapshot saved: {embed_path}")

# ✅ Entry point
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    train()
