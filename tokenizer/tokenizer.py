# tokenizer.py
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

# 1. Load training text
DATA_PATH = "data/corpus.txt"  # <-- your raw text file
assert os.path.exists(DATA_PATH), "Input training data not found!"

# 2. Create a new tokenizer object with BPE model
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# 3. Use basic whitespace splitter before BPE
tokenizer.pre_tokenizer = Whitespace()

# 4. Define the training config
trainer = BpeTrainer(
    vocab_size=1000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

# 5. Train the tokenizer on your corpus
tokenizer.train(files=[DATA_PATH], trainer=trainer)

# 6. Save tokenizer in GPT-compatible format
os.makedirs("tokenizer", exist_ok=True)
tokenizer.save("tokenizer/tokenizer.json")

# Optional: extract vocab and merges (for compatibility)
tokenizer.model.save("tokenizer/")  # saves vocab.json and merges.txt

print("✅ Tokenizer trained and saved to 'tokenizer/' folder.")


# ======== GPTTokenizer class for usage (encode/decode) ========

class GPTTokenizer:
    """
    Wrapper class to use the trained BPE tokenizer for encoding and decoding text.
    """
    def __init__(self, tokenizer_path="tokenizer/tokenizer.json"):
        # Load the trained tokenizer from file
        assert os.path.exists(tokenizer_path), "Tokenizer file not found!"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        """
        Encode a string into a list of token IDs.
        Example: "Hello" → [123, 456]
        """
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into text.
        Example: [123, 456] → "Hello"
        """
        return self.tokenizer.decode(token_ids)
