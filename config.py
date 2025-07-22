# config.py

# Vocabulary (GPT-2 tokenizer)
vocab_size = 50257  # ✅ matches GPT-2 tokenizer

# Model size (adjusted for this task)
embed_dim = 64           # Slightly increased from 32 for better expressiveness
max_seq_len = 32         # Keep small for short greeting turns
num_heads = 4            # Better attention spread
ff_dim = 128             # Adjusted for new embed_dim
num_layers = 4           # Slightly deeper network

# Training
batch_size = 4           # Small batch helps with generalization
learning_rate = 5e-4     # Slightly higher to encourage faster convergence
epochs = 10              # Should be enough, early stopping can help

# Paths
tokenizer_path = "tokenizer/tokenizer.json"  # not used by code, but reference
corpus_path = "data/corpus.txt"
checkpoint_path = "checkpoints/minigpt.pt"
