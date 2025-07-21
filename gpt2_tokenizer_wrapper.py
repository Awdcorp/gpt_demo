# gpt2_tokenizer_wrapper.py
from transformers import GPT2TokenizerFast

class GPT2TokenizerWrapper:
    def __init__(self, tokenizer_name="gpt2"):
        self.tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Ensure pad exists

    def encode(self, text):
        return self.tokenizer.encode(text, truncation=True, max_length=32)

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
