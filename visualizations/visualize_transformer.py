# visualizations/visualize_transformer.py

import streamlit as st
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append("model")
sys.path.append("tokenizer")

from tokenizer import GPTTokenizer
from model.embedding import GPTEmbedding
from model.block import TransformerBlock
import config

def run():
    st.header("🧠 Transformer Block Visualization")

    # Input text
    text = st.text_input("Enter input text:", "The meaning of life is 42")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []

    if not token_ids:
        st.warning("Please enter some text.")
        return

    token_strs = [tokenizer.tokenizer.id_to_token(id) for id in token_ids]
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    st.write(f"Token IDs: {token_ids}")
    st.write(f"Tokens: {token_strs}")

    # Embedding + Transformer
    embed_layer = GPTEmbedding(config.vocab_size, config.embed_dim, config.max_seq_len)
    block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim)

    with torch.no_grad():
        embedded = embed_layer(token_tensor)
        output = block(embedded)

    # === Attention Weights ===
    if hasattr(block.attn, "attn_weights"):
        st.subheader("🧭 Multi-Head Self-Attention Weights")
        attn_weights = block.attn.attn_weights.squeeze(0)  # (heads, T, T)

        for h, matrix in enumerate(attn_weights):
            st.markdown(f"**Head {h+1}**")
            fig, ax = plt.subplots()
            sns.heatmap(matrix[:len(token_strs), :len(token_strs)],
                        xticklabels=token_strs,
                        yticklabels=token_strs,
                        cmap="viridis", square=True, ax=ax,
                        cbar_kws={"label": "Attention Score"})
            ax.set_xlabel("Key Tokens")
            ax.set_ylabel("Query Tokens")
            st.pyplot(fig)
    else:
        st.warning("⚠️ Attention weights not captured. Did you patch `attention.py` to store `self.attn_weights`?")

    # === Feedforward Output ===
    st.subheader("⚡ FeedForward Output Embeddings")
    ff_output = output.squeeze(0).numpy()

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(ff_output, cmap="plasma", xticklabels=False, yticklabels=token_strs, ax=ax,
                cbar_kws={"label": "Activation"})
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Token")
    st.pyplot(fig)
