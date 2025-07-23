# visualize_logits_softmax.py

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import plotly.express as px
from gpt2_tokenizer_wrapper import GPT2TokenizerWrapper
from model.gpt_model import MiniGPT
from config import *
import random

def sample_token(probs, top_k=10, top_p=1.0, temperature=1.0):
    log = []  # for explanation block

    # Apply temperature scaling
    scaled_logits = torch.log(probs + 1e-9) / temperature
    log.append(f"Applied temperature {temperature:.2f} to logits")
    scaled_probs = F.softmax(scaled_logits, dim=0)

    # Apply top-k filtering
    if top_k < len(scaled_probs):
        topk_vals, topk_idxs = torch.topk(scaled_probs, top_k)
        mask = torch.zeros_like(scaled_probs)
        mask[topk_idxs] = scaled_probs[topk_idxs]
        log.append(f"Top-k filtering: kept top {top_k} tokens")
        scaled_probs = mask / mask.sum()  # re-normalize

    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(scaled_probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        cutoff = cumulative_probs > top_p
        if torch.any(cutoff):
            last_valid = (cutoff == True).nonzero(as_tuple=True)[0][0].item() + 1
            valid_indices = sorted_indices[:last_valid]
            mask = torch.zeros_like(scaled_probs)
            mask[valid_indices] = scaled_probs[valid_indices]
            log.append(f"Top-p filtering: retained top {last_valid} tokens (p={top_p:.2f})")
            scaled_probs = mask / mask.sum()  # re-normalize

    # Sample from adjusted distribution
    sampled_index = torch.multinomial(scaled_probs, 1).item()
    sampled_prob = scaled_probs[sampled_index].item()
    log.append(f"Sampled token ID {sampled_index} from filtered distribution")

    return sampled_index, sampled_prob, scaled_probs, log

def run():
    st.set_page_config(page_title="📈 Logits & Softmax Visualizer", layout="wide")
    st.title("📈 Final Prediction: Logits & Softmax")

    st.markdown("""
    This module shows how the final prediction is made in a GPT model by:
    - Viewing the **raw logits** (unnormalized scores)
    - Applying **Softmax** to get probabilities
    - Visualizing the **Top-k token candidates**
    - Sampling using **temperature**, **top-k**, and **top-p** filtering
    - Showing the **final predicted token**
    """)

    # Input prompt
    prompt = st.text_input("📝 Enter prompt:", value="hello how are")
    top_k_display = st.slider("📊 Top-K Tokens to Display", min_value=5, max_value=20, value=10)

    # Sampling parameters
    st.sidebar.header("🎛️ Sampling Settings")
    top_k = st.sidebar.slider("Top-K", min_value=1, max_value=50, value=10)
    top_p = st.sidebar.slider("Top-P (Nucleus)", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
    sampling_method = st.sidebar.selectbox("Prediction Strategy", ["Greedy (Argmax)", "Sampled (Top-k/p)"])

    # Load tokenizer and model
    tokenizer = GPT2TokenizerWrapper()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if prompt:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(input_tensor)[0]  # shape: [1, seq_len, vocab_size]

        final_logits = logits[0, -1, :]
        softmax_probs = F.softmax(final_logits, dim=0)

        # Top-k for display
        display_values, display_indices = torch.topk(softmax_probs, k=top_k_display)
        display_tokens = [tokenizer.decode([idx.item()]) for idx in display_indices]
        display_logits = [final_logits[idx].item() for idx in display_indices]
        display_probs = [val.item() for val in display_values]

        # Display logits
        st.subheader("🔢 Final Logits (Top-k)")
        df_logits = pd.DataFrame({"Token": display_tokens, "Logit": display_logits})
        fig1 = px.bar(df_logits, x="Token", y="Logit", color="Logit", title="Top-k Logits (Before Softmax)")
        st.plotly_chart(fig1, use_container_width=True)

        # Display probabilities
        st.subheader("📊 Softmax Probabilities (Top-k)")
        df_probs = pd.DataFrame({"Token": display_tokens, "Probability": display_probs})
        fig2 = px.bar(df_probs, x="Token", y="Probability", color="Probability", title="Top-k Probabilities (After Softmax)")
        st.plotly_chart(fig2, use_container_width=True)

        # Sampling
        if sampling_method == "Greedy (Argmax)":
            predicted_index = torch.argmax(softmax_probs).item()
            predicted_prob = softmax_probs[predicted_index].item()
            method_note = "Argmax of softmax distribution"
            explanation = ["Selected the token with highest probability (greedy method)"]
        else:
            predicted_index, predicted_prob, _, explanation = sample_token(
                softmax_probs, top_k, top_p, temperature)
            method_note = f"Top-k={top_k}, Top-p={top_p}, Temperature={temperature}"

        predicted_token = tokenizer.decode([predicted_index])

        st.success(f"🎯 **Predicted Token:** `{predicted_token}` with probability **{predicted_prob:.4f}**")
        st.caption(f"(Token ID: {predicted_index}, Method: {method_note})")

        # Explanation block
        st.markdown("""
        ### 🧠 Sampling Explanation Log
        These are the exact steps used during the sampling process:
        """)
        for step in explanation:
            st.markdown(f"- {step}")
