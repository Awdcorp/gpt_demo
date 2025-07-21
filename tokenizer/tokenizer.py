# tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
import os

# Load training text
DATA_PATH = "data/corpus.txt"
assert os.path.exists(DATA_PATH), "❌ Input training data not found!"

# Create BPE tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# ✅ Normalize: lowercase + remove accents
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])

# ✅ Pre-tokenize by whitespace
tokenizer.pre_tokenizer = Whitespace()

# ✅ Trainer config for tiny dataset
trainer = BpeTrainer(
    vocab_size=100,  # Reduced from 200
    min_frequency=1,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]"]
)

# ✅ Train tokenizer
tokenizer.train(files=[DATA_PATH], trainer=trainer)

# ✅ Save
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save("tokenizer/tokenizer.json")
tokenizer.model.save("tokenizer/")  # saves vocab.json and merges.txt

print("✅ Tokenizer trained and saved to 'tokenizer/' folder.")

# ======== GPTTokenizer class for encoding/decoding =========

class GPTTokenizer:
    """
    Wrapper for the trained BPE tokenizer.
    """
    def __init__(self, tokenizer_path="tokenizer/tokenizer.json"):
        assert os.path.exists(tokenizer_path), "Tokenizer file not found!"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids)
