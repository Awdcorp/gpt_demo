import sys
import os

# 🚨 Dynamically add root path of your project (e.g., D:/gpt_demo/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import torch
import json
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *
import streamlit as st



# === Load model & tokenizer ===
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
model.eval()

tokenizer = GPTTokenizer(tokenizer_path)

# === Input prompt ===
prompt = "hello world"
input_ids = tokenizer.encode(prompt)
input_tensor = torch.tensor([input_ids])

# === Get vectors with trace
with torch.no_grad():
    logits, token_vectors, attn_weights = model(input_tensor, return_vectors=True, return_attn=True)

# === Load trace data ===
with open("visualizations_3d/assets/trace.json", "r") as f:
    trace = json.load(f)

token_labels = trace["input_tokens"]
embedding_vectors = np.array(trace["token_vectors"]["embedding"][0])  # (T, D)
after_attn_vectors = np.array(trace["token_vectors"]["after_attn"][0][0])  # layer 0
after_ffn_vectors = np.array(trace["token_vectors"]["after_ffn"][0][0])    # layer 0
attn_weights = trace["attn_weights"]["layer_0"]["head_0"][0]  # (T, T)

# === Stack all vectors for PCA ===
T = len(token_labels)
# === Clean and prepare token vectors ===
embedding_vectors = token_vectors["embedding"][0].cpu().numpy()             # (T, D)
attn_layers = [v[0].cpu().numpy() for v in token_vectors["after_attn"]]     # List[(T, D)]
ffn_layers = [v[0].cpu().numpy() for v in token_vectors["after_ffn"]]       # List[(T, D)]

# === Combine all steps: [embedding, attn_layer1, ..., ffn_layer1, ...]
steps_list = [embedding_vectors] + attn_layers + ffn_layers                 # List of (T, D)

# === Verify consistent shapes
D = embedding_vectors.shape[1]
assert all(s.shape == (embedding_vectors.shape[0], D) for s in steps_list), "Mismatch in shape!"

# === Stack to shape: (T, S, D)
steps_array = np.stack(steps_list, axis=1)  # (T, S, D)
T, S, D = steps_array.shape

# === PCA projection
flat_vectors = steps_array.reshape(-1, D)   # (T×S, D)
pca = PCA(n_components=3)
pca_vectors = pca.fit_transform(flat_vectors)
pca_steps = pca_vectors.reshape(T, S, 3)    # (T, S, 3)

# === Plotly 3D plot setup ===
fig = go.Figure()

# === Colors by token ===
colors = px.colors.qualitative.Bold
color_map = {tok: colors[i % len(colors)] for i, tok in enumerate(token_labels)}

# === Token trace paths ===
for i, token in enumerate(token_labels):
    path = pca_steps[i]  # (3, 3)
    fig.add_trace(go.Scatter3d(
        x=path[:, 0], y=path[:, 1], z=path[:, 2],
        mode='lines+markers+text',
        name=token,
        line=dict(color=color_map[token], width=4),
        marker=dict(size=4),
        text=[f"{token}<br>step {j}" for j in range(3)],
        hoverinfo="text"
    ))

# === Attention arrows (layer 0, head 0) ===
for src in range(T):
    for tgt in range(T):
        attn_weights = np.array(trace["attn_weights"]["layer_0"]["head_0"]) 
        weight = attn_weights[src][tgt]  # ✅ fetch float value
        if weight > 0.05:  # ✅ now this works correctly
            src_pos = pca_steps[src, 1]  # after attention
            tgt_pos = pca_steps[tgt, 1]
            fig.add_trace(go.Scatter3d(
                x=[src_pos[0], tgt_pos[0]],
                y=[src_pos[1], tgt_pos[1]],
                z=[src_pos[2], tgt_pos[2]],
                mode='lines',
                line=dict(color='gray', width=weight * 10),
                hoverinfo='none',
                showlegend=False
            ))
# === Layout ===
fig.update_layout(
    title="🧠 Token Vector Flow with Attention (Layer 0, Head 0)",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3",
    ),
    margin=dict(l=0, r=0, t=40, b=0)
)

# === Streamlit page ===
st.set_page_config(page_title="🧠 LLM 3D Visualizer", layout="wide")
st.title("🧠 Token Vector Flow (Embedding → Attention → FFN)")

st.plotly_chart(fig, use_container_width=True)
