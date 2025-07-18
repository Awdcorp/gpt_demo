# visualizations/visualize_full_flow.py

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
    st.title("🔁 Full MiniGPT Pipeline Visualization")

    text = st.text_input("Enter input prompt:", "The meaning of life is 42")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []

    if not token_ids:
        st.warning("Please enter valid text input.")
        return

    token_strs = [tokenizer.tokenizer.id_to_token(id) for id in token_ids]
    token_tensor = torch.tensor(token_ids).unsqueeze(0)  # shape: (1, seq_len)

    st.markdown("## 1️⃣ Tokenization")
    st.code(f"Input Text: {text}")
    st.write(f"Tokens: {token_strs}")
    st.write(f"Token IDs: {token_ids}")

    # Embedding
    embed_layer = GPTEmbedding(config.vocab_size, config.embed_dim, config.max_seq_len)
    with torch.no_grad():
        embeddings = embed_layer(token_tensor)

    st.markdown("## 2️⃣ Embedding Layer Output")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    sns.heatmap(embeddings[0].numpy(), cmap="magma", xticklabels=False, yticklabels=token_strs, ax=ax1, cbar_kws={"label": "Embedding Value"})
    ax1.set_xlabel("Embedding Dim")
    ax1.set_ylabel("Tokens")
    st.pyplot(fig1)

    # Transformer
    block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim)
    with torch.no_grad():
        transformer_out = block(embeddings)

    st.markdown("## 3️⃣ Transformer Block Output")
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    sns.heatmap(transformer_out[0].numpy(), cmap="cividis", xticklabels=False, yticklabels=token_strs, ax=ax2, cbar_kws={"label": "Post-Attn+FF Value"})
    ax2.set_xlabel("Embedding Dim")
    ax2.set_ylabel("Tokens")
    st.pyplot(fig2)

    # Attention Weights (if available)
    if hasattr(block.attn, "attn_weights"):
        st.markdown("## 4️⃣ Attention Weights (Head 1)")
        attn_matrix = block.attn.attn_weights[0][0].numpy()
        fig3, ax3 = plt.subplots()
        sns.heatmap(attn_matrix[:len(token_strs), :len(token_strs)],
                    xticklabels=token_strs,
                    yticklabels=token_strs,
                    cmap="viridis", ax=ax3,
                    cbar_kws={"label": "Attention Score"})
        ax3.set_xlabel("Key Tokens")
        ax3.set_ylabel("Query Tokens")
        st.pyplot(fig3)
    else:
        st.warning("⚠️ Attention weights not found. Did you patch `attention.py` to store `self.attn_weights`?")

    st.success("✅ Full pipeline visual complete.")
