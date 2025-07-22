import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import torch.nn.functional as F
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

# Setup
st.set_page_config(page_title="🔁 Token Pipeline Visualizer", layout="wide")
st.title("🔁 Full Token Lifecycle Visualizer")

# Load tokenizer & model
tokenizer = GPTTokenizer(tokenizer_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# --- Input Prompt ---
prompt = st.text_input("💬 Enter input prompt:", value="Hello, how are you?")
layer_idx = st.slider("📚 Layer to visualize", 0, num_layers-1, 0)
top_k = st.slider("🎯 Top-K for logits", 5, 20, 10)

if prompt:
    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(device)
    tokens = [tokenizer.decode([i]) for i in input_ids]
    st.markdown("### 🧾 Tokenized Input")
    st.write(pd.DataFrame({"Index": list(range(len(tokens))), "Token": tokens, "Token ID": input_ids}))

    token_idx = st.slider("🔢 Select Token to Trace", 0, len(tokens)-1, 0)
    selected_token = tokens[token_idx]
    st.info(f"Tracing token `{selected_token}` at position {token_idx}")

    # Forward Pass
    with torch.no_grad():
        logits, token_vectors, attn_weights = model(
            input_tensor, return_attn=True, return_vectors=True
        )

    # Extract stages
    embed_vec = token_vectors['embedding'][0, token_idx].cpu().numpy()
    attn_vec = token_vectors['after_attn'][layer_idx][0, token_idx].cpu().numpy()
    ffn_vec = token_vectors['after_ffn'][layer_idx][0, token_idx].cpu().numpy()
    final_logits = logits[0, -1]
    softmax_probs = F.softmax(final_logits, dim=0)

    # --- Embedding ---
    st.subheader("🔢 Token Embedding Vector")
    df_embed = pd.DataFrame({"Dimension": list(range(embed_dim)), "Value": embed_vec})
    st.plotly_chart(px.line(df_embed, x="Dimension", y="Value", title="Embedding Vector"))

    # --- After Attention ---
    st.subheader("🧠 After Attention")
    df_attn = pd.DataFrame({
        "Dimension": list(range(embed_dim)),
        "Before Attention": embed_vec,
        "After Attention": attn_vec
    })
    st.plotly_chart(px.line(df_attn, x="Dimension", y=["Before Attention", "After Attention"], title="Attention Transformation"))

    # --- After Feedforward ---
    st.subheader("🔬 After Feedforward")
    df_ffn = pd.DataFrame({
        "Dimension": list(range(embed_dim)),
        "Post-Attention": attn_vec,
        "Post-FFN": ffn_vec
    })
    st.plotly_chart(px.line(df_ffn, x="Dimension", y=["Post-Attention", "Post-FFN"], title="Feedforward Transformation"))

    # --- Final Logits & Prediction ---
    st.subheader("📈 Final Logits and Prediction")
    top_vals, top_idxs = torch.topk(softmax_probs, k=top_k)
    top_tokens = [tokenizer.decode([i.item()]) for i in top_idxs]
    df_logits = pd.DataFrame({
        "Token": top_tokens,
        "Probability": [v.item() for v in top_vals]
    })
    st.plotly_chart(px.bar(df_logits, x="Token", y="Probability", title="Top-k Predicted Tokens"))

    best_token = tokenizer.decode([torch.argmax(softmax_probs).item()])
    best_prob = torch.max(softmax_probs).item()
    st.success(f"🎯 Predicted Token: `{best_token}` with Probability: {best_prob:.4f}")

    # --- Summary ---
    st.subheader("🔁 Full Token Vector Evolution")
    df_summary = pd.DataFrame({
        "Dimension": list(range(embed_dim)),
        "Embedding": embed_vec,
        "After Attention": attn_vec,
        "After FFN": ffn_vec
    })
    st.plotly_chart(px.line(df_summary, x="Dimension", y=["Embedding", "After Attention", "After FFN"], title="Full Vector Lifecycle"))
