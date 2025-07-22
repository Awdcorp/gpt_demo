import streamlit as st
import torch
import numpy as np
import plotly.express as px
import pandas as pd
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

def run():
    st.set_page_config(page_title="🧠 Attention Visualizer", layout="wide")
    st.title("🔎 Transformer Attention Visualizer")

    with st.expander("🧠 What is Attention? (Click to expand)", expanded=False):
        st.markdown("""
        In GPT, **attention** lets each token decide which other tokens to focus on.
        This is done through matrices of attention scores between tokens.

        - Each **head** learns to focus on different relationships.
        - The output is a set of **attention heatmaps**, showing which tokens "attended to" which.

        This tool shows attention for each token in a sentence — across layers and heads.
        """)

    tokenizer = GPTTokenizer(tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    sentence = st.text_input("💬 Enter a sentence to visualize attention:", value="The cat sat on the mat.")

    if sentence:
        token_ids = tokenizer.encode(sentence)
        decoded_tokens = [tokenizer.decode([i]) for i in token_ids]
        tokens = [f"{tok}_{idx}" for idx, tok in enumerate(decoded_tokens)]
        input_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

        with torch.no_grad():
            _, _, attn_weights = model(input_tensor, return_attn=True)  # [Layers][B, H, T, T]

        num_attn_layers = len(attn_weights)
        layer = st.slider("📚 Select Layer", 0, num_attn_layers - 1, 0)
        head = st.slider("🧠 Select Head", 0, num_heads - 1, 0)

        attn = attn_weights[layer][0, head].cpu().numpy()  # [T, T]

        # 🎯 Highlight strongest attention per row with emoji
        text_matrix = []
        for i, row in enumerate(attn):
            max_j = np.argmax(row)
            formatted_row = []
            for j, val in enumerate(row):
                val_str = f"{val:.2f}"
                if j == max_j:
                    val_str = f"🎯 {val_str}"
                formatted_row.append(val_str)
            text_matrix.append(formatted_row)

        df = pd.DataFrame(attn, columns=tokens, index=tokens)
        fig1 = px.imshow(df, color_continuous_scale="Blues")
        fig1.update_traces(
            text=text_matrix,
            texttemplate="%{text}",
            hovertemplate="From %{y} to %{x}<br>Score: %{z:.2f}<extra></extra>"
        )
        fig1.update_layout(width=600, height=600)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"### 🎯 Layer {layer} | Head {head}")
            st.plotly_chart(fig1, use_container_width=True)

        # Optional comparison
        if st.checkbox("🔁 Compare with another head?", value=False):
            layer2 = st.slider("📚 Compare: Select Layer", 0, num_attn_layers - 1, 1, key="layer2")
            head2 = st.slider("🧠 Compare: Select Head", 0, num_heads - 1, 1, key="head2")
            attn2 = attn_weights[layer2][0, head2].cpu().numpy()

            text_matrix2 = []
            for i, row in enumerate(attn2):
                max_j = np.argmax(row)
                formatted_row = []
                for j, val in enumerate(row):
                    val_str = f"{val:.2f}"
                    if j == max_j:
                        val_str = f"🎯 {val_str}"
                    formatted_row.append(val_str)
                text_matrix2.append(formatted_row)

            df2 = pd.DataFrame(attn2, columns=tokens, index=tokens)
            fig2 = px.imshow(df2, color_continuous_scale="Blues")
            fig2.update_traces(
                text=text_matrix2,
                texttemplate="%{text}",
                hovertemplate="From %{y} to %{x}<br>Score: %{z:.2f}<extra></extra>"
            )
            fig2.update_layout(width=600, height=600)

            with col2:
                st.markdown(f"### 🎯 Layer {layer2} | Head {head2}")
                st.plotly_chart(fig2, use_container_width=True)

        # ✅ Side-by-Side Comparison Table (interactive)
        st.markdown("### 🧩 Token-to-Token Attention Summary (Top Link Per Token)")

        summary_data = []
        for i, row in enumerate(attn):
            max_j = np.argmax(row)
            from_tok = tokens[i]
            to_tok = tokens[max_j]
            strength = row[max_j]
            summary_data.append({
                "🔹 From Token": from_tok,
                "🎯 Attends To": to_tok,
                "📊 Strength": round(strength, 2)
            })

        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, height=350)

        st.markdown("""
        #### 📘 What this heatmap shows:
        - Each **row** corresponds to a token in your sentence.
        - Each **cell** indicates how strongly that token attends to others.
        - 🎯 highlights the strongest attention in each row.

        For example:
        - If the `sat` row has 🎯 under `cat`, it means "sat" is most influenced by "cat".
        - This visualizes which words influence each other during prediction.

        ✅ You can explore how attention changes across different **layers** and **heads**.
        """)

    st.success("✅ Attention visualizer loaded successfully.")

if __name__ == "__main__":
    run()
