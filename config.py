# === Model Architecture ===
vocab_size = 1000
max_seq_len = 32
embed_dim = 64
num_heads = 4
ff_dim = 256
num_layers = 4
dropout = 0.1

# === Training Hyperparameters ===
batch_size = 8
learning_rate = 3e-4
epochs = 5

# === Paths ===
tokenizer_path = "tokenizer/tokenizer.json"
corpus_path = "data/corpus.txt"
checkpoint_path = "checkpoints/minigpt.pt"
