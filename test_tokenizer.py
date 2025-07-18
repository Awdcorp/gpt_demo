# test_tokenizer.py
from tokenizers import Tokenizer

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Try encoding
text = "Hello, how are you?"
encoded = tokenizer.encode(text)
print("🔠 Input text:", text)
print("🔢 Token IDs:", encoded.ids)
print("🧩 Tokens:", encoded.tokens)

# Try decoding
decoded = tokenizer.decode(encoded.ids)
print("📝 Decoded back:", decoded)
