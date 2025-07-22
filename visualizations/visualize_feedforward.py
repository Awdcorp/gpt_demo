import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

def run():
    st.set_page_config(page_title="🔬 Feedforward Layer Visualizer", layout="wide")
    st.title("🔬 Transformer Feedforward Layer Visualizer")

    with st.expander("📘 What is the Feedforward Layer?", expanded=False):
        st.markdown("""
        After attention, each token vector passes through a **feedforward network (FFN)**:

        1. Linear layer (projects to higher dimension)
        2. GELU activation (non-linear transformation)
        3. Linear layer (projects back to embedding dimension)

        This helps the model learn complex transformations **token-wise**, independently.
        """)

    # Load tokenizer and model
    tokenizer = GPTTokenizer(tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Input
    sentence = st.text_input("💬 Enter a sentence to visualize FFN effect:", value="The cat sat on the mat.")

    if sentence:
        token_ids = tokenizer.encode(sentence)
        tokens = [tokenizer.decode([i]) for i in token_ids]
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        # User controls
        selected_layer = st.slider("📚 Select Transformer Layer", 0, num_layers - 1, 0)
        token_idx = st.slider("🔢 Select Token Position", 0, len(tokens)-1, 0)

        # Forward pass until selected layer
        with torch.no_grad():
            x = model.backbone.embedding(input_tensor)
            for i in range(num_layers):
                block = model.backbone.blocks[i]

                normed = block.ln1(x)
                attn_out, _ = block.attn(normed, return_attn=True)
                x = x + attn_out

                normed_ff = block.ln2(x)
                ffn_in = normed_ff[0].cpu().numpy()
                ffn_out = block.ff(normed_ff)[0].cpu().numpy()
                x = x + block.ff(normed_ff)

                if i == selected_layer:
                    # Save pipeline vectors at selected layer
                    embed_vec = model.backbone.embedding(input_tensor)[0].cpu().numpy()
                    attn_vec = normed[0].cpu().numpy()
                    ffn_vec = x[0].cpu().numpy()
                    break

        # 🧠 FFN Transformation Plot
        st.markdown(f"### 📊 FFN Transformation for Token: `{tokens[token_idx]}`")

        vec_in = ffn_in[token_idx]
        vec_out = ffn_out[token_idx]

        df_ffn = pd.DataFrame({
            "Component": [f"dim_{i}" for i in range(embed_dim)],
            "Before FFN": vec_in[:embed_dim],
            "After FFN": vec_out[:embed_dim]
        })

        fig_ffn = px.line(df_ffn, x="Component", y=["Before FFN", "After FFN"], markers=True)
        fig_ffn.update_layout(width=900, height=400)
        st.plotly_chart(fig_ffn, use_container_width=True)

        st.markdown("""
        #### 🔍 What this shows:
        - The FFN applies a token-wise transformation.
        - The residual path adds it back to the original.
        - Observe how dimensions change token meaning post-attention.
        """)

        # 🔄 Pipeline View (embedding → attention → FFN)
        with st.expander("🔄 Full Pipeline View per Token", expanded=False):
            st.markdown(f"### 🔎 Transformation of Token: `{tokens[token_idx]}`")
            vec_embed = embed_vec[token_idx]
            vec_attn = attn_vec[token_idx]
            vec_ffn = ffn_vec[token_idx]

            df_pipe = pd.DataFrame({
                "Component": [f"dim_{i}" for i in range(embed_dim)],
                "Embedding": vec_embed[:embed_dim],
                "Post-Attention": vec_attn[:embed_dim],
                "Post-FFN": vec_ffn[:embed_dim]
            })

            fig_pipe = px.line(df_pipe, x="Component", y=["Embedding", "Post-Attention", "Post-FFN"], markers=True)
            fig_pipe.update_layout(width=900, height=400)
            st.plotly_chart(fig_pipe, use_container_width=True)

            st.markdown("""
            #### 🧠 What this shows:
            - See how each token vector evolves:
              1. Embedding (static meaning)
              2. After Attention (context added)
              3. After FFN (nonlinear reshaping)
            - This is the core of GPT's reasoning pipeline.
            """)

    st.success("✅ Feedforward visualizer loaded successfully.")

if __name__ == "__main__":
    run()
