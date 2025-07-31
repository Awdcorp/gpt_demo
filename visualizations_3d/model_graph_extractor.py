import json
from model.gpt_model import MiniGPT

# === Model Configuration (same as your config.py) ===
vocab_size = 3000
embed_dim = 64
max_seq_len = 32
num_heads = 4
ff_dim = 128
num_layers = 4
dropout = 0.1

# === Instantiate the model ===
model = MiniGPT(vocab_size, embed_dim, max_seq_len, num_heads, ff_dim, num_layers, dropout)

# === Build nodes and edges ===
nodes = []
edges = []

# Embedding Layer
nodes.append({
    "id": "embedding",
    "type": "Embedding",
    "params": {
        "embed_dim": embed_dim,
        "max_seq_len": max_seq_len
    }
})

# Transformer Blocks
for i in range(num_layers):
    block_id = f"block_{i}"
    nodes.append({
        "id": block_id,
        "type": "TransformerBlock",
        "params": {
            "index": i,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "ff_dim": ff_dim
        }
    })
    # Add edge from previous block or embedding
    source = "embedding" if i == 0 else f"block_{i-1}"
    edges.append({
        "source": source,
        "target": block_id
    })

# Final LayerNorm
nodes.append({
    "id": "norm",
    "type": "LayerNorm",
    "params": {
        "embed_dim": embed_dim
    }
})
edges.append({
    "source": f"block_{num_layers - 1}",
    "target": "norm"
})

# LM Head
nodes.append({
    "id": "lm_head",
    "type": "Linear",
    "params": {
        "output_dim": vocab_size
    }
})
edges.append({
    "source": "norm",
    "target": "lm_head"
})

# === Save to JSON ===
graph = {
    "nodes": nodes,
    "edges": edges
}

# Create output folder if needed
import os
os.makedirs("visualizations_3d/assets", exist_ok=True)

with open("visualizations_3d/assets/graph.json", "w") as f:
    json.dump(graph, f, indent=2)

print("✅ graph.json saved to visualizations_3d/assets/")
