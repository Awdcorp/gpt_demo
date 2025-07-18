import json
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from collections import Counter
import os

# ----------- Config -----------
TOKENIZER_PATH = "./tokenizer/tokenizer.json"  # Adjust if needed
CORPUS_FILE = "./data/corpus.txt"  # Raw corpus used during BPE training
TOP_K = 50  # Top tokens to show in histogram
# -------------------------------


def load_tokenizer(path):
    return Tokenizer.from_file(path)

def compute_token_frequencies(tokenizer, corpus_file):
    """Tokenize entire corpus and count token frequencies"""
    with open(corpus_file, "r", encoding="utf-8") as f:
        text = f.read()

    output = tokenizer.encode(text)
    tokens = output.tokens
    return Counter(tokens)

def plot_token_histogram(freqs, top_k=50):
    top_tokens = freqs.most_common(top_k)
    tokens, counts = zip(*top_tokens)

    plt.figure(figsize=(12, 6))
    plt.bar(tokens, counts)
    plt.xticks(rotation=90)
    plt.title(f"Top {top_k} Tokens by Frequency")
    plt.xlabel("Token")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def interactive_token_viewer(tokenizer):
    while True:
        inp = input("\nEnter a sentence to view BPE tokens (or 'exit'): ")
        if inp.strip().lower() == "exit":
            break
        encoded = tokenizer.encode(inp)
        print(f"Tokens: {encoded.tokens}")
        print(f"Token IDs: {encoded.ids}")

def main():
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer not found at {TOKENIZER_PATH}")
        return

    print("📦 Loading tokenizer...")
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    print("📊 Computing token frequency histogram...")
    if not os.path.exists(CORPUS_FILE):
        print(f"Corpus file not found: {CORPUS_FILE}")
    else:
        freqs = compute_token_frequencies(tokenizer, CORPUS_FILE)
        plot_token_histogram(freqs, TOP_K)

    print("🔍 Enter sentence to tokenize interactively:")
    interactive_token_viewer(tokenizer)

if __name__ == "__main__":
    main()
