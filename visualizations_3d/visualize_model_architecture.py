import sys
import os
import torch
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add root path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tokenizer.tokenizer import GPTTokenizer
from model.gpt_model import MiniGPT
from config import *

# === Streamlit setup ===
st.set_page_config(page_title="🧠 Model Architecture Visualizer", layout="wide")
st.title("🧠 GPT Architecture Diagram (MiniGPT)")

# === Define block layout in 3D ===
blocks = [
    ("Input Tokens", (0, 0, 0)),
    ("Embedding", (1, 0, 0)),
    ("Position Embedding", (1, 1, 0)),
]

for layer in range(num_layers):
    x = 2 + layer * 3
    blocks += [
        (f"LayerNorm_{layer}_1", (x, 0, 0)),
        (f"QKV_{layer}", (x + 1, 0, 0)),
        (f"Attention_{layer}", (x + 2, 0, 0)),
        (f"Add_Residual_{layer}_1", (x + 2, -1, 0)),
        (f"LayerNorm_{layer}_2", (x + 2.5, 0, 1)),
        (f"FFN_{layer}", (x + 3, 0, 1)),
        (f"Add_Residual_{layer}_2", (x + 3, -1, 1)),
    ]

last_x = 2 + (num_layers - 1) * 3 + 3
blocks += [
    ("Final LayerNorm", (last_x + 1, 0, 0)),
    ("Linear (LM Head)", (last_x + 2, 0, 0)),
    ("Softmax", (last_x + 3, 0, 0)),
    ("Predicted Token", (last_x + 4, 0, 0)),
]

# === Build plot ===
fig = go.Figure()

for name, (x, y, z) in blocks:
    fig.add_trace(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers+text',
        marker=dict(size=10, color='skyblue'),
        text=[name],
        textposition="top center",
        name=name,
        hoverinfo="text"
    ))

# === Connect arrows ===
def draw_arrow(start, end, color='gray'):
    fig.add_trace(go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode='lines',
        line=dict(color=color, width=4),
        showlegend=False,
        hoverinfo='none'
    ))

# === Draw logical arrows ===
for i in range(len(blocks) - 1):
    draw_arrow(blocks[i][1], blocks[i + 1][1])

# === Final layout ===
fig.update_layout(
    title="🧠 MiniGPT Model Architecture",
    scene=dict(
        xaxis=dict(title='Layer Axis'),
        yaxis=dict(title='Component Axis'),
        zaxis=dict(title='Depth Axis'),
    ),
    margin=dict(l=0, r=0, t=50, b=0)
)

st.plotly_chart(fig, use_container_width=True)
