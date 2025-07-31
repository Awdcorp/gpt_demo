# visualize_llm_3d.py

import json
import os
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

# === Load trace ===
trace_path = "visualizations_3d/assets/trace.json"
if not os.path.exists(trace_path):
    st.error("❌ trace.json not found. Please run generate_withlogs.py first.")
    st.stop()

with open(trace_path, "r") as f:
    trace = json.load(f)

tokens = trace["input_tokens"]
embedding = np.array(trace["token_vectors"]["embedding"])             # (T, D)
after_attn = np.array(trace["token_vectors"]["after_attn"])           # (L, T, D)
after_ffn = np.array(trace["token_vectors"]["after_ffn"])             # (L, T, D)

num_layers = after_attn.shape[0]
seq_len = len(tokens)

# === Stack all vectors per token ===
# Each token will have a path: [embedding, after_attn[0], after_ffn[0], ..., after_ffn[n]]
token_paths = []

for t in range(seq_len):
    path = [embedding[t]]  # Start with embedding
    for l in range(num_layers):
        path.append(after_attn[l][t])  # After attention
        path.append(after_ffn[l][t])   # After FFN
    token_paths.append(np.array(path))  # (1 + 2*L, D)

# === Flatten all points for PCA
all_points = np.vstack(token_paths)
pca = PCA(n_components=3)
points_3d = pca.fit_transform(all_points)

# === Map back to token paths in 3D
token_paths_3d = []
idx = 0
for path in token_paths:
    count = path.shape[0]
    token_paths_3d.append(points_3d[idx:idx+count])
    idx += count

# === Plotly 3D paths
fig = go.Figure()

colors = px.colors.qualitative.Bold
for i, path in enumerate(token_paths_3d):
    fig.add_trace(go.Scatter3d(
        x=path[:, 0],
        y=path[:, 1],
        z=path[:, 2],
        mode="lines+markers+text",
        name=tokens[i],
        text=[f"{tokens[i]}<br>Step {j}" for j in range(len(path))],
        line=dict(color=colors[i % len(colors)], width=4),
        marker=dict(size=4)
    ))

fig.update_layout(
    title="🧠 Token Vector Flow (Embedding → Attention → FFN)",
    scene=dict(
        xaxis_title="PC1", yaxis_title="PC2", zaxis_title="PC3"
    ),
    height=700
)

# === Streamlit UI
st.set_page_config(page_title="🧠 3D Token Flow Visualizer", layout="wide")
st.title("🧠 3D MiniGPT Token Flow Visualizer")
st.markdown(f"**Prompt:** `{trace['input_prompt']}`")
st.plotly_chart(fig, use_container_width=True)
