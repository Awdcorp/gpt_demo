import os
import json
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
from tokenizers import Tokenizer

# ==== Load Files ====
TOKENIZER_PATH = "../tokenizer/tokenizer.json"
MERGES_PATH = "../tokenizer/merges.txt"
DATA_PATH = "../data/corpus.txt"
VOCAB_PATH = "../tokenizer/vocab.json"

assert os.path.exists(TOKENIZER_PATH), "❌ tokenizer.json not found!"
assert os.path.exists(MERGES_PATH), "❌ merges.txt not found!"
assert os.path.exists(DATA_PATH), "❌ corpus.txt not found!"
assert os.path.exists(VOCAB_PATH), "❌ vocab.json not found!"

tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

with open(MERGES_PATH, "r", encoding="utf-8") as f:
    merges = f.read().splitlines()[1:]

with open(DATA_PATH, "r", encoding="utf-8") as f:
    full_corpus = f.read()

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)
id_to_token = {v: k for k, v in vocab.items()}

tokens_in_vocab = set(vocab.keys())
base_sentence = "Hello, how are you?"
encoded = tokenizer.encode(base_sentence)

# ==== UI Layout ====
st.set_page_config(page_title="Token Lifecycle", layout="wide")
st.title("🧬 Token Lifecycle Visualizer")

view = st.sidebar.radio("📌 Choose a View", [
    "Token Creation", 
    "Encoding", 
    "Decoding", 
    "🧬 Token Merge Graph"
])

# ---------------- Token Creation ----------------
if view == "Token Creation":
    st.header("🧱 Token Creation from BPE Merges")
    steps = st.slider("🔄 Merge Step", 1, len(merges), 1)
    merge_history = merges[:steps]

    tokens = set()
    merge_pairs = []

    for i, line in enumerate(merge_history):
        parts = line.strip().split()
        if len(parts) == 2:
            a, b = parts
            merge_pairs.append((a, b, f"{a}{b}", i))
            tokens.update([a, b, f"{a}{b}"])

    fig = go.Figure()
    for i, (a, b, merged, z) in enumerate(merge_pairs):
        fig.add_trace(go.Scatter3d(
            x=[i, i, i], y=[0, 1, 2], z=[z]*3,
            mode='text',
            text=[a, "+", b],
            textposition="top center",
            name=f"Merge {i+1}"
        ))
        fig.add_trace(go.Scatter3d(
            x=[i], y=[1], z=[z+1],
            mode='text',
            text=[merged],
            textposition="top center",
            name=f"Token {merged}"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Step',
            yaxis_title='Position',
            zaxis_title='Merge Layer'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Encoding ----------------
elif view == "Encoding":
    st.header("🔤 Encoding: Convert Sentence to Tokens")
    st.write(f"**Input Sentence**: `{base_sentence}`")

    x_vals, y_vals, z_vals, texts = [], [], [], []
    for i, token in enumerate(encoded.tokens):
        x_vals.append(i)
        y_vals.append(encoded.ids[i])
        z_vals.append(1)
        texts.append(token)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers+text',
        text=texts,
        textposition='top center',
        marker=dict(size=8)
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Token Position',
            yaxis_title='Token ID',
            zaxis_title='Layer'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------- Decoding ----------------
elif view == "Decoding":
    st.header("🔁 Decoding: Convert Tokens to Sentence")
    st.write(f"**Token IDs**: `{encoded.ids}`")

    decoded_tokens = [id_to_token.get(str(id_), "[UNK]") for id_ in encoded.ids]
    reconstructed = "".join(decoded_tokens).replace("▁", " ").strip()

    st.markdown(f"**Reconstructed Sentence**: `{reconstructed}`")

    st.markdown("---")
    st.subheader("🔍 Token ID to Text Mapping")
    for tid, text in zip(encoded.ids, decoded_tokens):
        st.write(f"`{tid}` → `{text}`")

# ---------------- Token Merge Graph ----------------
elif view == "🧬 Token Merge Graph":
    st.header("🧠 Token Merge Ancestry Graph")

    max_step = st.slider("🌀 Show merges up to step", 1, len(merges), 20)

    token_graph = nx.DiGraph()
    merge_steps = []

    for step_num, line in enumerate(merges[:max_step]):
        parts = line.strip().split()
        if len(parts) != 2:
            continue
        a, b = parts
        merged = a + b
        token_graph.add_edge(a, merged, step=step_num)
        token_graph.add_edge(b, merged, step=step_num)
        merge_steps.append((a, b, merged, step_num))

    # All tokens to show (merged + from vocab)
    tokens_from_merges = {m[2] for m in merge_steps}
    tokens_used = set(encoded.tokens)
    all_vocab_tokens = set(vocab.keys())
    tokens_to_visualize = tokens_from_merges.union(all_vocab_tokens)

    # Classify each token's origin
    token_origin = {}
    for token in tokens_to_visualize:
        if token in tokens_used:
            token_origin[token] = "used"
        elif token in tokens_from_merges:
            token_origin[token] = "merged"
        else:
            token_origin[token] = "base"

    # Build edge list
    edges_to_draw = []
    for token in tokens_to_visualize:
        if token in token_graph:
            for pred in token_graph.predecessors(token):
                edges_to_draw.append((pred, token))

    G = token_graph.edge_subgraph(edges_to_draw)
    pos = nx.spring_layout(G, seed=42, dim=3)

    x_nodes, y_nodes, z_nodes, labels, colors = [], [], [], [], []
    for node, (x, y, z) in pos.items():
        x_nodes.append(x)
        y_nodes.append(y)
        z_nodes.append(z)
        labels.append(node)
        if token_origin.get(node) == "used":
            colors.append("red")
        elif token_origin.get(node) == "merged":
            colors.append("blue")
        else:
            colors.append("yellow")

    x_edges, y_edges, z_edges = [], [], []
    for src, dst in G.edges():
        x_edges += [pos[src][0], pos[dst][0], None]
        y_edges += [pos[src][1], pos[dst][1], None]
        z_edges += [pos[src][2], pos[dst][2], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=x_edges, y=y_edges, z=z_edges,
        mode='lines',
        line=dict(color='gray', width=2),
        name='Merge Path'
    ))

    fig.add_trace(go.Scatter3d(
        x=x_nodes, y=y_nodes, z=z_nodes,
        mode='markers+text',
        text=labels,
        textposition='top center',
        marker=dict(size=6, color=colors),
        name='Tokens'
    ))

    fig.update_layout(
        title=f"🧬 Token Merge Lineage (Steps 1–{max_step})",
        scene=dict(
            xaxis_title='Token Merge Influence (X)',
            yaxis_title='Semantic Grouping (Y)',
            zaxis_title='Merge Depth (Z)'
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=700,
        legend=dict(
            itemsizing='constant',
            traceorder="normal",
            title="Legend"
        )
    )

    st.plotly_chart(fig, use_container_width=True)

