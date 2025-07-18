# visualizations/visualize_embedding_projection.py

import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import sys
sys.path.append("model")
sys.path.append("tokenizer")

from tokenizer import GPTTokenizer
from model.embedding import GPTEmbedding
import config

def run():
    st.title("📈 Token Embedding Projection: PCA / t-SNE")

    text = st.text_input("Enter input text:", "The meaning of life is 42")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []

    if not token_ids:
        st.warning("Please enter some text to visualize.")
        return

    tokens = [tokenizer.tokenizer.id_to_token(id) for id in token_ids]

    # Load embeddings
    embed_layer = GPTEmbedding(config.vocab_size, config.embed_dim, config.max_seq_len)
    token_tensor = torch.tensor(token_ids).unsqueeze(0)  # (1, seq_len)
    with torch.no_grad():
        embeddings = embed_layer.token_embedding(token_tensor)[0].numpy()  # (seq_len, embed_dim)

    # Projection method
    method = st.radio("Projection Method", ["PCA", "t-SNE"])
    if method == "PCA":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=5, n_iter=1000)

    reduced = reducer.fit_transform(embeddings)

    # DataFrame for plotting
    df = pd.DataFrame(reduced, columns=["x", "y"])
    df["Token"] = tokens

    # Plot
    st.subheader("🔍 2D Embedding Projection")
    fig, ax = plt.subplots()
    ax.scatter(df["x"], df["y"], s=100)

    for i, row in df.iterrows():
        ax.text(row["x"] + 0.1, row["y"], row["Token"], fontsize=12)

    ax.set_title(f"Projection using {method}")
    st.pyplot(fig)
