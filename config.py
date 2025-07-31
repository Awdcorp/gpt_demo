# === Config that matches checkpoint ===

vocab_size = 10000          # ✅ Match training tokenizer
embed_dim = 64              # (unchanged)
max_seq_len = 64            # ✅ Match checkpoint
num_heads = 4               # (unchanged)
ff_dim = 256                # ✅ Match checkpoint
num_layers = 4              # (unchanged)
dropout = 0.1

# Paths
tokenizer_path = "tokenizer/tokenizer.json"
corpus_path = "data/corpus.txt"
checkpoint_path = "checkpoints/minigpt_e1_b2000.pt"
