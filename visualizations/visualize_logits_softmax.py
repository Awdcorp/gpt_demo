# visualize_logits_softmax.py

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

def run():
    st.set_page_config(page_title="📈 Logits & Softmax Visualizer", layout="wide")
    st.title("📈 Final Prediction: Logits & Softmax")

    st.markdown("""
    This module shows how the final prediction is made in a GPT model by:
    - Viewing the **raw logits** (unnormalized scores)
    - Applying **Softmax** to get probabilities
    - Visualizing the **Top-k token candidates**
    - Showing the **final predicted token**
    """)

    # Input prompt
    prompt = st.text_input("📝 Enter prompt:", value="hello how are")
    top_k = st.slider("🎯 Top-K Tokens to Display", min_value=5, max_value=20, value=10)

    # Load tokenizer and model
    tokenizer = GPTTokenizer(tokenizer_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if prompt:
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        # Forward pass to get logits
        with torch.no_grad():
            logits = model(input_tensor)[0]  # shape: [1, seq_len, vocab_size]

        # Focus on final token position
        final_logits = logits[0, -1, :]  # shape: [vocab_size]
        softmax_probs = F.softmax(final_logits, dim=0)

        # Get Top-k tokens
        topk_values, topk_indices = torch.topk(softmax_probs, k=top_k)
        topk_tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
        topk_logits = [final_logits[idx].item() for idx in topk_indices]
        topk_probs = [val.item() for val in topk_values]

        # Display results
        st.subheader("🔢 Final Logits (Top-k)")
        df_logits = pd.DataFrame({
            "Token": topk_tokens,
            "Logit": topk_logits
        })
        fig1 = px.bar(df_logits, x="Token", y="Logit", color="Logit", title="Top-k Logits (Before Softmax)")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("📊 Softmax Probabilities (Top-k)")
        df_probs = pd.DataFrame({
            "Token": topk_tokens,
            "Probability": topk_probs
        })
        fig2 = px.bar(df_probs, x="Token", y="Probability", color="Probability", title="Top-k Probabilities (After Softmax)")
        st.plotly_chart(fig2, use_container_width=True)

        # Show final prediction
        predicted_index = torch.argmax(softmax_probs).item()
        predicted_token = tokenizer.decode([predicted_index])
        predicted_prob = softmax_probs[predicted_index].item()

        st.success(f"🎯 **Predicted Token:** `{predicted_token}` with probability **{predicted_prob:.4f}**")
        st.caption(f"(Token ID: {predicted_index})")

