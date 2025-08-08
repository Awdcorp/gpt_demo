from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFD, StripAccents, Sequence
import os
import datetime
import json

# ======== GPTTokenizer class with encode/decode tracing =========
class GPTTokenizer:
    """
    Wrapper around a trained Hugging Face `tokenizers` BPE tokenizer.
    Adds optional tracing for encode/decode:
      - encode(): original text, normalized text, pre-tokens, final tokens + ids
      - decode(): input ids, decoded text
    You can inject a logger function (e.g., your existing `log`) so traces
    go into the same generation .log file.
    """

    def __init__(self, tokenizer_path="tokenizer/tokenizer.json", trace=False, logger=None, log_to_file=False, log_file_path=None):
        # Ensure the tokenizer file exists
        assert os.path.exists(tokenizer_path), "Tokenizer file not found!"
        self.tokenizer = Tokenizer.from_file(tokenizer_path)

        # Tracing controls
        self.trace = trace
        self.logger = logger  # callable(str) -> None
        self.log_to_file = log_to_file
        # Default file if file logging is requested but no path supplied
        if self.log_to_file and not log_file_path:
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file_path = f"tokenizer_encode_decode_{ts}.log"
        self.log_file_path = log_file_path

    def _log(self, msg: str):
        """Internal logger that prefers injected logger, then file, then print."""
        if self.logger is not None:
            self.logger(msg)
        else:
            if self.log_to_file and self.log_file_path:
                with open(self.log_file_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            print(msg)

    def set_logger(self, logger_fn):
        """Allow wiring in an external logger later (e.g., your `log()` from generation)."""
        self.logger = logger_fn

    def encode(self, text: str, trace: bool | None = None):
        """
        Convert input text to a list of token IDs.
        When tracing, also logs normalization output, pre-tokenization result,
        final tokens and ids.
        """
        do_trace = self.trace if trace is None else trace

        # (1) Original text
        if do_trace:
            self._log(f"📝 Original text: {text}")

        # (2) Normalization (if configured on the tokenizer)
        normalized_text = text
        try:
            if getattr(self.tokenizer, "normalizer", None) is not None:
                normalized_text = self.tokenizer.normalizer.normalize_str(text)
        except Exception:
            # If HF internals change, fall back silently
            pass

        if do_trace:
            self._log(f"🔧 Normalized: {normalized_text}")

        # (3) Pre-tokenization preview (if configured)
        pre_tokens_preview = None
        try:
            if getattr(self.tokenizer, "pre_tokenizer", None) is not None:
                # pre_tokenize_str takes (input, output_list); we collect pairs
                pre_tokens = []
                self.tokenizer.pre_tokenizer.pre_tokenize_str(normalized_text, pre_tokens)
                # HF returns a list of tuples: (piece, (start, end))
                pre_tokens_preview = [t[0] for t in pre_tokens]
        except Exception:
            pre_tokens_preview = None

        if do_trace and pre_tokens_preview is not None:
            self._log(f"✂️ Pre-tokens: {pre_tokens_preview}")

        # (4) Final encoding (this also applies BPE merges)
        enc = self.tokenizer.encode(text)
        ids = enc.ids
        toks = enc.tokens

        if do_trace:
            self._log(f"🧩 Tokens: {toks}")
            self._log(f"🔢 Token IDs: {ids}")

        return ids

    def decode(self, token_ids: list[int], trace: bool | None = None):
        """
        Convert a list of token IDs back to text.
        When tracing, logs the input IDs and the decoded text.
        """
        do_trace = self.trace if trace is None else trace

        if do_trace:
            self._log(f"🔢 Input Token IDs: {token_ids}")

        text = self.tokenizer.decode(token_ids)

        if do_trace:
            self._log(f"🧾 Decoded Text: {text}")

        return text


if __name__ == "__main__":
    # === Prepare log file ===
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"tokenizer_training_{timestamp}.log"
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    # === Load training corpus ===
    DATA_PATH = "data/corpus.txt"
    assert os.path.exists(DATA_PATH), "❌ Input training data not found!"

    file_size = os.path.getsize(DATA_PATH)
    num_lines = sum(1 for _ in open(DATA_PATH, encoding="utf-8"))

    log(f"📂 Loading corpus: {DATA_PATH}")
    log(f"📏 Corpus size: {file_size} bytes, {num_lines} lines\n")

    with open(DATA_PATH, encoding="utf-8") as f:
        sample_lines = [next(f).strip() for _ in range(min(5, num_lines))]
    log("📄 Corpus sample lines:")
    for line in sample_lines:
        log(f"   {line}")
    log("")

    # === Initialize a BPE tokenizer model ===
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))  # Define UNK token for unknowns

    # === Normalization: Lowercase + remove accents (Unicode NFD normalization) ===
    tokenizer.normalizer = Sequence([
        NFD(),         # Decompose Unicode characters (e.g., é → e + ́)
        StripAccents() # Remove any accent characters
    ])
    log("🔧 Normalization: Using NFD + StripAccents\n")

    norm_sample = "Héllo HOW Are Ü?"
    norm_output = tokenizer.normalizer.normalize_str(norm_sample)
    log(f"🔧 Normalization Example:\n   Before: {norm_sample}\n   After:  {norm_output}\n")

    # === Pre-tokenizer: split input on whitespace ===
    tokenizer.pre_tokenizer = Whitespace()
    log("✂️ Pre-tokenizer: Whitespace split\n")

    pre_tok_sample = "Hello how are you?"
    pre_tokens = tokenizer.pre_tokenizer.pre_tokenize_str(pre_tok_sample)
    log(f"✂️ Pre-tokenization Example:\n   Input:  {pre_tok_sample}\n   Output: {pre_tokens}\n")

    # === Define training configuration for BPE ===
    trainer = BpeTrainer(
        vocab_size=1000,  # Target vocabulary size
        min_frequency=1,  # Minimum frequency of a token to be included
        special_tokens=["[PAD]", "[UNK]", "[EOS]", "[SYS]", "[USR]", "[AST]", "[EOT]"]  #  add chat-style special tokens for real-model SFT compatibility
    )
    log("⚙️ Training Configuration:")
    log(f"   Vocab size: {trainer.vocab_size}")
    log(f"   Min frequency: {trainer.min_frequency}")
    log(f"   Special tokens: {trainer.special_tokens}\n")

    # === Train the tokenizer on the corpus ===
    log("🚀 Starting tokenizer training...")
    tokenizer.train(files=[DATA_PATH], trainer=trainer)
    log("✅ Tokenizer training complete.\n")

    # === Save trained tokenizer ===
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/tokenizer.json")
    tokenizer.model.save("tokenizer/")
    log("💾 Tokenizer saved to 'tokenizer/' folder\n")

    # === Live Merge Step Logging from merges.txt ===
    merges_path = os.path.join("tokenizer", "merges.txt")
    if os.path.exists(merges_path):
        with open(merges_path, encoding="utf-8") as f:
            merges = [line.strip() for line in f if not line.startswith("#") and line.strip()]
        log(f"🔀 Merge steps recorded: {len(merges)} total")
        for i, merge in enumerate(merges):
            log(f"   Step {i+1}: {merge}")
        log("")

    # === Final vocab logging ===
    vocab_path = os.path.join("tokenizer", "vocab.json")
    if os.path.exists(vocab_path):
        with open(vocab_path, encoding="utf-8") as f:
            vocab = json.load(f)
        log(f"📚 Final vocab size: {len(vocab)}")
        for token, idx in list(vocab.items())[:50]:  # First 50 entries
            log(f"   ID {idx}: '{token}'")
        if len(vocab) > 50:
            log("   ... (vocab list truncated)")
        log("")

    # === Save training log ===
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"📝 Full training log saved to: {log_file_path}")
