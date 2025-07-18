# visualizations/visualize_logits.py

import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append("model")
sys.path.append("tokenizer")

from tokenizer import GPTTokenizer
from model.embedding import GPTEmbedding
from model.block import TransformerBlock
import config

class OutputHead(nn.Module):
    def __init__(self, embed_dim, vocab_size):
        super().__init__()
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        return self.lm_head(x)

def run():
    st.title("🧠 Output Prediction and Logits")

    text = st.text_input("Input sequence:", "The meaning of life is")
    tokenizer = GPTTokenizer(config.tokenizer_path)
    token_ids = tokenizer.encode(text) if text.strip() else []

    if not token_ids:
        st.warning("Please enter a valid prompt.")
        return

    tokens = [tokenizer.tokenizer.id_to_token(i) for i in token_ids]
    token_tensor = torch.tensor(token_ids).unsqueeze(0)

    embed_layer = GPTEmbedding(config.vocab_size, config.embed_dim, config.max_seq_len)
    block = TransformerBlock(config.embed_dim, config.num_heads, config.ff_dim)
    output_head = OutputHead(config.embed_dim, config.vocab_size)

    with torch.no_grad():
        emb = embed_layer(token_tensor)
        out = block(emb)
        logits = output_head(out)  # (1, seq_len, vocab)
        probs = torch.softmax(logits[0, -1], dim=0)  # Only for last token

    # Get top-k predictions
    topk = 10
    top_probs, top_ids = torch.topk(probs, k=topk)
    top_tokens = [tokenizer.tokenizer.id_to_token(i.item()) for i in top_ids]

    # Prepare DataFrame
    df = pd.DataFrame({
        "Token": top_tokens,
        "Probability": top_probs.numpy()
    })

    st.subheader("🔮 Top 10 Predictions for Next Token")
    fig, ax = plt.subplots()
    sns.barplot(data=df, y="Token", x="Probability", ax=ax)
    ax.set_title("Next Token Prediction")
    st.pyplot(fig)

    st.success(f"Most likely next token: `{top_tokens[0]}`")
