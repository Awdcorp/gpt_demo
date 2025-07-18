# visualizations/visualize_embeddings.py

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
import config

def run():
    st.header("🧬 Embedding Layer Visualization")

    # Input
    text = st.text_input("Enter input text:", "The meaning of life is 42")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []
    token_strs = [tokenizer.tokenizer.id_to_token(id) for id in token_ids]

    st.write(f"Token IDs: {token_ids}")
    st.write(f"Tokens: {token_strs}")

    if not token_ids:
        st.warning("Please enter some text to visualize embeddings.")
        return

    # Convert to tensor
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    # Init embedding model
    embed_layer = GPTEmbedding(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len,
    )

    # Load from checkpoint (optional)
    if os.path.exists(config.checkpoint_path):
        state_dict = torch.load(config.checkpoint_path, map_location="cpu")
        if "embedding.token_embedding.weight" in state_dict:
            embed_layer.load_state_dict({
                "token_embedding.weight": state_dict["embedding.token_embedding.weight"],
                "position_embedding.weight": state_dict["embedding.position_embedding.weight"]
            }, strict=False)
            st.success("✅ Loaded embedding weights from checkpoint.")
        else:
            st.warning("⚠️ Embedding weights not found. Using random init.")
    else:
        st.warning("⚠️ Checkpoint not found. Using random embedding weights.")

    # Forward pass
    with torch.no_grad():
        embeddings = embed_layer(token_tensor)[0].numpy()

    # Visualize
    st.subheader("📊 Embedding Heatmap (Token × Dimension)")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(embeddings, cmap="viridis", xticklabels=False, yticklabels=token_strs, ax=ax, cbar_kws={"label": "Embedding Value"})
    ax.set_xlabel("Embedding Dimension")
    ax.set_ylabel("Token")
    st.pyplot(fig)
