from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
import os

# ======== GPTTokenizer class for encoding/decoding =========

class GPTTokenizer:
    """
    Wrapper class for handling tokenization using a trained BPE tokenizer.
    This class is used during inference to encode/decode text.
    """
    def __init__(self, tokenizer_path="tokenizer/tokenizer.json"):
        # Ensure the tokenizer file exists before loading
        assert os.path.exists(tokenizer_path), "Tokenizer file not found!"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

    def encode(self, text):
        """
        Convert input text to a list of token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids):
        """
        Convert a list of token IDs back to text.
        """
        return self.tokenizer.decode(token_ids)

# ======== Training logic: only runs if file is executed directly =========

if __name__ == "__main__":
    # === Load training corpus ===
    DATA_PATH = "data/corpus.txt"
    assert os.path.exists(DATA_PATH), "❌ Input training data not found!"

    # === Initialize a BPE tokenizer model ===
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # Define UNK token for unknowns

    # === Normalization: Lowercase + remove accents (Unicode NFD normalization) ===
    tokenizer.normalizer = Sequence([
        NFD(),         # Decompose Unicode characters (e.g., é → e + ́)
        StripAccents() # Remove any accent characters
    ])

    # === Pre-tokenizer: split input on whitespace ===
    tokenizer.pre_tokenizer = Whitespace()

    # === Define training configuration for BPE ===
    trainer = BpeTrainer(
        vocab_size=1000,  # Target vocabulary size
        min_frequency=1,  # Minimum frequency of a token to be included
        special_tokens=["[PAD]", "[UNK]"]  # Add standard special tokens
    )

    # === Train the tokenizer on the corpus ===
    tokenizer.train(files=[DATA_PATH], trainer=trainer)

    # === Save trained tokenizer ===
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/tokenizer.json")      # Save complete tokenizer config
    tokenizer.model.save("tokenizer/")              # Save vocab.json and merges.txt separately

    print("✅ Tokenizer trained and saved to 'tokenizer/' folder.")
