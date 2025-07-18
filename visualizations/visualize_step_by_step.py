# visualizations/visualize_step_by_step.py

import streamlit as st
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("model")
sys.path.append("tokenizer")

from tokenizer import GPTTokenizer
from model.embedding import GPTEmbedding
from model.block import TransformerBlock
import config

def visualize_tensor(tensor, token_labels, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(tensor, cmap="plasma", xticklabels=False, yticklabels=token_labels, ax=ax, cbar_kws={"label": "Activation"})
    ax.set_title(title)
    st.pyplot(fig)

def run():
    st.title("🎥 Step-by-Step Transformer Flow")

    if "step" not in st.session_state:
        st.session_state.step = 0

    text = st.text_input("Input Text:", "The meaning of life is 42")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []

    if not token_ids:
        st.warning("Please enter some input text.")
        return

    tokens = [tokenizer.tokenizer.id_to_token(i) for i in token_ids]
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    embed_layer = GPTEmbedding(config.vocab_size, config.embed_dim, config.max_seq_len)
    block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim)

    with torch.no_grad():
        embeddings = embed_layer(token_tensor)        # Step 1
        norm1 = block.ln1(embeddings)                 # Step 2
        attn_out = block.attn(norm1)                  # Step 3
        res1 = embeddings + attn_out                  # Step 4
        norm2 = block.ln2(res1)                       # Step 5
        ff_out = block.ff(norm2)                      # Step 6
        final_out = res1 + ff_out                     # Step 7

    all_steps = [
        ("Token Embeddings", embeddings[0].numpy()),
        ("LayerNorm 1", norm1[0].numpy()),
        ("Self-Attention Output", attn_out[0].numpy()),
        ("Residual Connection 1", res1[0].numpy()),
        ("LayerNorm 2", norm2[0].numpy()),
        ("FeedForward Output", ff_out[0].numpy()),
        ("Final Output", final_out[0].numpy()),
    ]

    # Sidebar Controls
    st.sidebar.markdown("## Step Controls")
    st.sidebar.markdown(f"**Current Step:** {st.session_state.step + 1} / {len(all_steps)}")

    if st.sidebar.button("⬅️ Back") and st.session_state.step > 0:
        st.session_state.step -= 1
    if st.sidebar.button("➡️ Next") and st.session_state.step < len(all_steps) - 1:
        st.session_state.step += 1

    step_title, tensor = all_steps[st.session_state.step]
    st.markdown(f"## Step {st.session_state.step + 1}: {step_title}")
    visualize_tensor(tensor, tokens, step_title)

    st.info(f"Showing output of: **{step_title}**")
