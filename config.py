# config.py

# Model architecture
vocab_size   = 10000  
embed_dim    = 64
max_seq_len  = 64
num_heads    = 4
ff_dim       = 256
num_layers   = 4

# Training
batch_size     = 8
learning_rate  = 1e-3
epochs         = 5

# Paths
tokenizer_path = "tokenizer/tokenizer.json"
corpus_path    = "data/corpus.txt"
checkpoint_path = "checkpoints/minigpt.pt"
